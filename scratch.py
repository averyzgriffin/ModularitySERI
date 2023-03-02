import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import ModularArithmeticDataset
from models import OrthogMLP, Transformer


device = torch.device("cuda:0")


def add_hooks(model, hookfuncs: list):
    model.handles = []
    for module in model.layers:
        for hook in hookfuncs:
            handle = module.register_forward_hook(hook)
            model.handles.append(handle)


def mlp():
    model = OrthogMLP(8, 6, 4, 2)
    # with torch.no_grad():
    #     model.fc0.weight = nn.Parameter(torch.ones_like(model.fc0.weight), requires_grad=True)
    #     model.fc1.weight = nn.Parameter(torch.tensor([[1., -2.], [3., -4.], [5., -6.]], requires_grad=True).T)
    #     model.fc2.weight = nn.Parameter(torch.tensor([[1., 1.]], requires_grad=True))

    model.to(device)
    add_hooks(model, hookfuncs=[model.compute_derivatives_hook])

    x = torch.randn((1, 1, 8), requires_grad=True)
    y = torch.tensor([42]).to(device)
    data = [(x, y)]

    def func(in_):
        return model.fc1(in_).relu()

    for b, (x, label) in enumerate(data):
        model.derivatives = []
        x = x.reshape(len(x), -1).to(device)

        hidden = model.fc0(x).relu()
        detached_hidden = hidden.detach()
        detached_hidden.requires_grad = True

        # out = model.fc1(detached_hidden).relu()

        # func = model.fc1()

        jacob = torch.autograd.functional.jacobian(func, hidden)

        # out.backward(torch.ones_like(out))

        grad = jacob.squeeze().to("cpu").numpy()
        grad_true = model.derivatives[1].squeeze().detach().to("cpu").numpy()

        shape = jacob.shape
        true_shape = grad_true.shape

        print()


def scratch_t(model, data):

    def func(in_):
        return model.blocks[0].attn(in_)

    # cache = {}
    # model.remove_all_hooks()
    # model.cache_all(cache, incl_bwd=True)

    df = torch.zeros((128,128)).to(device)
    all_jacobs = []
    all_outs = []
    for b, (x, label) in enumerate(data):
        if b == 5:
            break
        x = x.reshape(len(x), -1).to(device)

        # Send through the embedding and detach
        hidden = model.pos_embed(model.embed(x))
        detached_hidden = hidden.detach()
        detached_hidden.requires_grad = True

        # out = model.blocks[0].attn(model.pos_embed(detached_hidden))  # Send through the attention block and stop

        # Jacobian method
        jacob = torch.autograd.functional.jacobian(func, hidden)
        new_jacob = (jacob.sum(dim=(2,5)).pow(2).reshape(jacob.shape[0], -1, jacob.shape[0], jacob.shape[-1]) / torch.tensor([2*3]).to(device)).sum(dim=(0,2))

        # Grad method
        # out.backward(torch.ones_like(out))
        # grad = detached_hidden.grad

        # Extra steps for the gradients - not essential atm
        # squared = torch.pow(jacob, 2)
        # summed_squared = torch.sum(squared, dim=1, keepdim=True)
        # summed_batch = torch.sum(summed_squared, dim=0, keepdim=True)
        df += new_jacob

        print(b)

    df /= b * data.batch_size
    print()


def manual_computation(model, data):
    cache = {}
    model.remove_all_hooks()
    model.cache_all(cache, incl_bwd=True)

    for b, (x, label) in enumerate(data):
        original_logits = model(x)[:, -1]

        which_layers = ["blocks.0.attn.hook_attn_softmax", "blocks.0.attn.hook_attn_pre_softmax",
                        "blocks.0.attn.hook_k", "blocks.0.attn.hook_q", "blocks.0.attn.hook_v"]
        act_dict = {k: v for (k, v) in cache.items() if k in which_layers}

        derivative = 3

        softmax = act_dict['blocks.0.attn.hook_attn_softmax']
        Wv = model.blocks[0].attn.W_V
        v = act_dict['blocks.0.attn.hook_v']
        k = act_dict['blocks.0.attn.hook_k']
        Wq = model.blocks[0].attn.W_Q
        q = act_dict['blocks.0.attn.hook_q']
        Wk = model.blocks[0].attn.W_K
        # softmax_derivative = derivative_softmax(act_dict['blocks.0.attn.hook_attn_pre_softmax'])
        softmax_shape = softmax.shape
        Wv_shape = Wv.shape
        v_shape = v.shape
        k_shape = k.shape
        Wq_shape = Wq.shape
        q_shape = q.shape
        Wk_shape = Wk.shape

        value = softmax * Wv + ( (v / torch.sqrt(torch.tensor([32]))) * 1 * (k * Wq + q * Wk) )

#     ( softmax_value * Wv + ( (v / torch.sqrt(torch.tensor([32]))) * softmax_derivative * (k * Wq + Q * Wk) ) )


def derivative_softmax(x):
    sm = torch.unsqueeze(F.softmax(x), dim=1)
    jacobian = torch.zeros(x.shape + x.shape[:-1])
    for i in range(x.shape[-2]):
        for j in range(x.shape[-1]):
            for k in range(x.shape[-2]):
                for l in range(x.shape[-1]):
                    if i == k and j == l:
                        jacobian[..., i, j, k, l] = sm[..., i, j] * (1 - sm[..., k, l])
                    else:
                        jacobian[..., i, j, k, l] = -sm[..., i, j] * sm[..., k, l]
    return jacobian


def test():
    x = torch.ones((1, 4), requires_grad=True)
    model = OrthogMLP(4, 3, 2, 1)
    with torch.no_grad():
        model.fc0.weight = nn.Parameter(torch.ones_like(model.fc0.weight))
        model.fc1.weight = nn.Parameter(torch.tensor([[1., -2.], [3., -4.], [5., -6.]], requires_grad=True).T)
        model.fc2.weight = nn.Parameter(torch.tensor([[1., 1.]], requires_grad=True))

    for module in model.layers:
        module.register_forward_hook(model.compute_derivatives_hook)

    prediction = model(x)
    prediction.backward()
    # Print the gradients

    # layer0_grad = torch.autograd.grad(prediction, activations[0], retain_graph=True, allow_unused=True)

    print()


def manual_computation(model, dataloader, layer, device):
    for b, (x, y) in enumerate(dataloader):
        x = x[0]
        x = x.reshape(len(x), -1).to(device)
        prediction = model(x)
        loss = torch.mean(prediction)
        loss.backward()
        grad1 = model.fc0.weight.grad
        grad2 = model.fc1.weight.grad
        grad3 = model.fc2.weight.grad
        e = 3
        # activations = model.activations
        # L = activations[layer]
        # L_1 = activations[layer+1]
        # gradients = torch.autograd.grad(L_1, L, create_graph=True, retain_graph=True)
        # g = gradients[0]
    return grad3


def compute_jacobian(model, dataloader, layer_index, device):
    jacobian = None
    for b, (x, y) in enumerate(dataloader):
        x = x.reshape(len(x), -1).to(device)

        # prediction = model(x.reshape(len(x), -1))

        # input_ = x
        # output = model.layers[layer_index](x)
        # input_.requiresGrad_(True)
        # output.backward(torch.ones_like(output), retain_graph=True)
        # jacobian = input_.grad

        # input_.grad = None

    return jacobian


def build_model(num_layers, d_vocab, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, use_ln):
    model = Transformer(num_layers=num_layers, d_vocab=d_vocab, d_model=d_model, d_mlp=d_mlp, d_head=d_head,
                        num_heads=num_heads, n_ctx=n_ctx, act_type=act_type, use_cache=False, use_ln=use_ln)
    return model


def load_model(model_dir, device):
    p = 113
    d_model = 128
    num_layers = 1
    d_vocab = p + 1
    n_ctx = 3
    d_mlp = 4 * d_model
    num_heads = 4
    assert d_model % num_heads == 0
    d_head = d_model // num_heads
    act_type = 'ReLU'
    use_ln = False

    run_saved_data = torch.load(model_dir)
    model = Transformer(num_layers=num_layers, d_vocab=d_vocab, d_model=d_model, d_mlp=d_mlp, d_head=d_head,
                        num_heads=num_heads, n_ctx=n_ctx, act_type=act_type, use_cache=False, use_ln=use_ln)
    model.to(device)
    model.load_state_dict(run_saved_data['model'])
    return model


if __name__ == '__main__':
    num_layers = 1
    p = 113
    d_model = 128
    num_heads = 4
    d_vocab = p + 1
    n_ctx = 3
    d_mlp = 4 * d_model
    assert d_model % num_heads == 0
    d_head = d_model // num_heads
    act_type = 'ReLU'  # ['ReLU', 'GeLU']
    use_ln = False
    path = r"C:\Users\Avery\Projects\ModularitySERI\saved_models\default101_final.pth"

    network = load_model(path, device)
    # network = build_model(num_layers, d_vocab, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, use_ln)

    fn_name = 'add'
    dataset = ModularArithmeticDataset(p, fn_name, device)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # mlp()
    scratch_t(network, dataloader)
    # manual_computation(network, dataloader)




# (10, 3, 5, 10, 3, 5)
# # sum across 3s while keeping heads separate
# [
#
#
#     [
#         [
#             [1, 2, 3, 4, 5],
#             [1, 2, 3, 4, 5], # all 0s
#             [1, 2, 3, 4, 5]  # all 0s
#         ],
#         [
#             [1, 2, 3, 4, 5],
#             [1, 2, 3, 4, 5], # all 0s
#             [1, 2, 3, 4, 5]  # all 0s
#         ],
#         [
#             [1, 2, 3, 4, 5],
#             [1, 2, 3, 4, 5], # all 0s
#             [1, 2, 3, 4, 5]  # all 0s
#         ],
#         [
#             [1, 2, 3, 4, 5],
#             [1, 2, 3, 4, 5], # all 0s
#             [1, 2, 3, 4, 5]  # all 0s
#         ],
#         [
#             [1, 2, 3, 4, 5],
#             [1, 2, 3, 4, 5], # all 0s
#             [1, 2, 3, 4, 5]  # all 0s
#         ]
#     ],
#
#
#
#
#     [
#         [
#             [1, 2, 3, 4, 5],
#             [1, 2, 3, 4, 5],
#             [1, 2, 3, 4, 5]  # all 0s
#         ],
#         [
#             [1, 2, 3, 4, 5],
#             [1, 2, 3, 4, 5],
#             [1, 2, 3, 4, 5] # all 0s
#         ],
#         [
#             [1, 2, 3, 4, 5],
#             [1, 2, 3, 4, 5],
#             [1, 2, 3, 4, 5] # all 0s
#         ],
#         [
#             [1, 2, 3, 4, 5],
#             [1, 2, 3, 4, 5],
#             [1, 2, 3, 4, 5] # all 0s
#         ],
#         [
#             [1, 2, 3, 4, 5],
#             [1, 2, 3, 4, 5],
#             [1, 2, 3, 4, 5] # all 0s
#         ]
#     ],
#
#
#
#
#     [
#         [
#             [1, 2, 3, 4, 5], # very similar to other 2 in this trio
#             [1, 2, 3, 4, 5], # very similar to other 2 in this trio
#             [1, 2, 3, 4, 5]  # very similar to other 2 in this trio
#         ],
#         [
#             [1, 2, 3, 4, 5], # very similar to other 2 in this trio
#             [1, 2, 3, 4, 5], # very similar to other 2 in this trio
#             [1, 2, 3, 4, 5]  # very similar to other 2 in this trio
#         ],
#         [
#             [1, 2, 3, 4, 5], # very similar to other 2 in this trio
#             [1, 2, 3, 4, 5], # very similar to other 2 in this trio
#             [1, 2, 3, 4, 5]  # very similar to other 2 in this trio
#         ],
#         [
#             [1, 2, 3, 4, 5], # very similar to other 2 in this trio
#             [1, 2, 3, 4, 5], # very similar to other 2 in this trio
#             [1, 2, 3, 4, 5]  # very similar to other 2 in this trio
#         ],
#         [
#             [1, 2, 3, 4, 5], # very similar to other 2 in this trio
#             [1, 2, 3, 4, 5], # very similar to other 2 in this trio
#             [1, 2, 3, 4, 5]  # very similar to other 2 in this trio
#         ]
#     ]
#
#
# ]
#
#
#


