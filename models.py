import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


def add_layers(model, dimensions):
    layers = []
    for i, (dimension) in enumerate(dimensions):
        fc = nn.Linear(dimension, dimensions[i + 1])
        setattr(model, f"fc{i}", fc)
        layers.append(fc)
        if len(dimensions) == i + 2:
            break

    return layers


class OrthogMLP(nn.Module):
    layers = []

    def __init__(self, *dimensions):
        super(OrthogMLP, self).__init__()

        self.layers = add_layers(self, dimensions)
        self.relu = nn.LeakyReLU()
        self.activations = []
        self.derivatives = []
        self.handles = []

    def forward(self, x):
        self.activations = []
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
            # x_and_bias = torch.cat((x,torch.ones(x.shape[0],1)), dim=1)
            # self.activations.append(x)

        x = self.layers[-1](x)
        self.activations.append(x)
        return x

    def compute_derivatives_hook(self, module, in_, out_):
        output_of_layer_L = out_
        mask = (output_of_layer_L > 0).float()
        weights_L_Lplus1 = module.weight
        # bias_L_Lplus1 = module.bias
        # all_edges = torch.cat((weights_L_Lplus1, bias_L_Lplus1.unsqueeze(dim=1)), dim=1)
        # masked_weights = mask * all_edges.T
        masked_weights = mask * weights_L_Lplus1.T
        self.derivatives.append(masked_weights)
        return out_

    def grab_activations_hook(self, module, in_, out_):
        # x_and_bias = torch.cat((in_[0],torch.ones(in_[0].shape[0],1)), dim=1)
        self.activations.append(in_[0])
        return out_

    def backward_hook(self, module, grad_input, grad_output):
        # Compute the derivative of the current layer with respect to the previous layer
        d = grad_output[0].T * module.weight
        self.derivatives[module] = d

        gt = grad_output[0].view(module.weight.shape[0], -1)
        derivative = gt.T @ module.weight
        return (derivative, None)

    def get_loss(self, dataloader, loss_fc, device):
        losses = []
        for b, (x, y) in enumerate(dataloader):
            x = x.to(device)
            prediction = self(x.reshape(len(x), -1))
            loss = loss_fc(prediction, y.to(device))
            losses.append(loss.item())
        return np.mean(losses)

    def get_accuracy(self, dataloader, device):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                prediction = self(x.reshape(len(x), -1))
                predicted = torch.argmax(prediction.data, dim=1)
                total += y.size(0)
                correct += (predicted == y.to(device)).sum().item()
        return correct / total

    def get_x_y_batches(self, dataloader, device):
        all_batches_x = []
        all_batches_y = []
        for b, (x, y) in enumerate(dataloader):
            x = x.reshape(len(x), -1).to(device)
            prediction = self(x)
            all_batches_x.append(x)
            all_batches_y.append(prediction)

        return torch.cat(all_batches_x, dim=0), torch.cat(all_batches_y, dim=0)


class SimpleMLP(nn.Module):
    layers = []

    def __init__(self, *dimensions):
        super(SimpleMLP, self).__init__()

        self.layers = add_layers(self, dimensions)

        self.relu = nn.LeakyReLU()

        self.activations_dict = dict()
        self.clusters = []
        self.m1_patterns, self.m2_patterns = [], []
        self.m1_counter = dict()
        self.m2_counter = dict()
        self.joint_counter = dict()

        self.avery = 0

    def forward(self, x):
        activation = [x]
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
            # x = (x>0.5).float()
            # activation.append((x > 0.3).float())# TODO taking this out for orthog decomposition

        # activation.append( (self.layers[-1](x) > 0.5).float() )# TODO taking this out for orthog decomposition
        # self.collect_activations(activation)  # TODO taking this out for orthog decomposition

        return self.layers[-1](x)

    def collect_activations(self, activation):
        flattened_activation = []
        for layer in activation[:-1]:
            flattened_activation += layer.squeeze().tolist()
        flattened_activation += activation[-1].tolist()[0]

        self.activations_dict[tuple(flattened_activation[:4])] = flattened_activation
        self.extract_modules(self.clusters, flattened_activation)

    def extract_modules(self, clusters, activation):
        m1_modules = []
        m2_modules = []

        # m1 = [x for c, x in zip(clusters, activation) if c == 0]
        # m2 = [x for c, x in zip(clusters, activation) if c == 1]
        # m1_modules.append(m1)
        # m2_modules.append(m2)

        m1 = activation[:4]
        m2 = [activation[-1]]
        m1_modules.append(m1)
        m2_modules.append(m2)

        self.update_dict_counter(tuple(m1), self.m1_counter)
        self.update_dict_counter(tuple(m2), self.m2_counter)
        self.update_dict_counter(tuple([tuple(m1),tuple(m2)]), self.joint_counter)

        # return self.get_unique(m1_modules), self.get_unique(m2_modules)

    @staticmethod
    def get_unique(lst):
        return [list(a) for a in set(tuple(b) for b in lst)]

    @staticmethod
    def update_dict_counter(key, d: dict):
        if key not in d:
            d[key] = 1
        else:
            d[key] += 1


"""
Transformers Classes - Grabbed from Neel's A Mechanistic Interpretability Analysis of Grokking -------
"""
# A helper class to get access to intermediate activations (inspired by Garcon)
# It's a dummy module that is the identity function by default
# I can wrap any intermediate activation in a HookPoint and get a convenient
# way to add PyTorch hooks
class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []

    def give_name(self, name):
        # Called by the model at initialisation
        self.name = name

    def add_hook(self, hook, dir='fwd'):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output,
        # which are the same for a HookPoint)
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)

        if dir == 'fwd':
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir == 'bwd':
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")

    def remove_hooks(self, dir='fwd'):
        if (dir == 'fwd') or (dir == 'both'):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir == 'bwd') or (dir == 'both'):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ['fwd', 'bwd', 'both']:
            raise ValueError(f"Invalid direction {dir}")

    def forward(self, x):
        return x


# Define network architecture
# I defined my own transformer from scratch so I'd fully understand each component
# - I expect this wasn't necessary or particularly important, and a bunch of this
# replicates existing PyTorch functionality

# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_model))

    def forward(self, x):
        # print("\nPre Embedding X: ", len(x), x[:5])
        x = torch.einsum('dbp -> bpd', self.W_E[:, x])
        # print("\nPost Embedding X: ", x.shape, x[:5])
        return x


class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_vocab))

    def forward(self, x):
        return (x @ self.W_U)


# Positional Embeddings
class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model) / np.sqrt(d_model))

    def forward(self, x):
        # print("\nPre Positional X: ", x.shape, x[:5])
        x = x + self.W_pos[:x.shape[-2]]
        # print("\nPost Positional X: ", x.shape, x[:5])
        return x


# LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon=1e-4, model=[None]):
        super().__init__()
        self.model = model
        self.w_ln = nn.Parameter(torch.ones(d_model))
        self.b_ln = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        if self.model[0].use_ln:
            x = x - x.mean(axis=-1)[..., None]
            x = x / (x.std(axis=-1)[..., None] + self.epsilon)
            x = x * self.w_ln
            x = x + self.b_ln
            return x
        else:
            return x


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx, model):
        super().__init__()
        self.model = model
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads) / np.sqrt(d_model))
        self.register_buffer('mask', torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()

    def forward(self, x):
        k = self.hook_k(torch.einsum('ihd,bpd->biph', self.W_K, x))
        q = self.hook_q(torch.einsum('ihd,bpd->biph', self.W_Q, x))
        v = self.hook_v(torch.einsum('ihd,bpd->biph', self.W_V, x))
        attn_scores_pre = torch.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = self.hook_attn(F.softmax(self.hook_attn_pre(attn_scores_masked / np.sqrt(self.d_head)), dim=-1))
        z = torch.einsum('biph,biqp->biqh', v, attn_matrix)
        z_flat = self.hook_z(einops.rearrange(z, 'b i q h -> b q (i h)'))
        out = torch.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out


# MLP Layers
class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        self.model = model
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model) / np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.act_type = act_type
        # self.ln = LayerNorm(d_mlp, model=self.model)
        self.hook_pre_relu = HookPoint()
        self.hook_post_hidden = HookPoint()
        assert act_type in ['ReLU', 'GeLU']

    def forward(self, x):
        x = self.hook_pre_relu(torch.einsum('md,bpd->bpm', self.W_in, x) + self.b_in)
        if self.act_type == 'ReLU':
            x = F.relu(x)
        elif self.act_type == 'GeLU':
            x = F.gelu(x)
        x = self.hook_post_hidden(x)
        x = torch.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model):
        super().__init__()
        self.model = model
        # self.ln1 = LayerNorm(d_model, model=self.model)
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, model=self.model)
        # self.ln2 = LayerNorm(d_model, model=self.model)
        self.mlp = MLP(d_model, d_mlp, act_type, model=self.model)
        self.hook_post_projz = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x):
        x = self.hook_resid_mid(x + self.hook_post_projz(self.attn((self.hook_resid_pre(x)))))
        x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp((x))))
        return x


# Full transformer
class Transformer(nn.Module):
    def __init__(self, num_layers, d_vocab, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, use_cache=False,
                 use_ln=True):
        super().__init__()
        self.cache = {}
        self.use_cache = use_cache

        self.embed = Embed(d_vocab, d_model)
        self.pos_embed = PosEmbed(n_ctx, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model=[self]) for i in
             range(num_layers)])
        # self.ln = LayerNorm(d_model, model=[self])
        self.unembed = Unembed(d_vocab, d_model)
        # self.use_ln = use_ln
        self.hook_embed_pre = HookPoint()
        self.hook_unembed_post = HookPoint()

        for name, module in self.named_modules():
            if type(module) == HookPoint:
                module.give_name(name)

    def forward(self, x):
        x = self.embed(self.hook_embed_pre(x))
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        # x = self.ln(x)
        x = self.hook_unembed_post(self.unembed(x))
        return x

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def hook_points(self):
        return [module for name, module in self.named_modules() if 'hook' in name]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks('fwd')
            hp.remove_hooks('bwd')

    def cache_all(self, cache, incl_bwd=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            cache[name] = tensor.detach()

        def save_hook_back(tensor, name):
            cache[name + '_grad'] = tensor[0].detach()

        for hp in self.hook_points():
            hp.add_hook(save_hook, 'fwd')
            if incl_bwd:
                hp.add_hook(save_hook_back, 'bwd')










