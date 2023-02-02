import os
import einops
import numpy as np
import torch
import torch.nn.functional as F

from models import Transformer
from analysis import interactive_histogram


def main(models: list, data, labels, which_models):
    num_heads = 4

    # W_O = einops.rearrange(model.blocks[0].attn.W_O, 'm (i h)->i m h', i=num_heads)
    # W_K = model.blocks[0].attn.W_K
    # W_Q = model.blocks[0].attn.W_Q
    # W_V = model.blocks[0].attn.W_V
    # W_in = model.blocks[0].mlp.W_in
    # W_out = model.blocks[0].mlp.W_out
    # W_pos = model.pos_embed.W_pos.T
    # # We remove the equals sign dimension from the Embed and Unembed, so we can
    # # apply a Fourier Transform over R^p
    # W_E = model.embed.W_E[:, :-1]
    # W_U = model.unembed.W_U[:, :-1].T
    #
    # # The initial value of the residual stream at position 2 - constant for all inputs
    # final_pos_resid_initial = model.embed.W_E[:, -1] + W_pos[:, 2]

    scores = []
    eigs = []
    num = 1
    for model in models:
        print("Model # ", num)
        num += 1
        # Create hooks
        cache = {}
        model.remove_all_hooks()
        model.cache_all(cache)

        # Forward Pass
        # data = data[:5]
        # labels = labels[:5]

        original_logits = model(data)[:, -1]
        original_logits = original_logits[:, :-1]

        # Get loss
        original_loss = cross_entropy_high_precision(original_logits, labels)
        score = original_loss.to("cpu").detach().float() # TODO + something

        # Check out particular hooks
        # attn_mat = cache['blocks.0.attn.hook_attn'][:, :, 2, :2]
        # neuron_acts = cache['blocks.0.mlp.hook_post'][:, -1]
        # neuron_acts_pre = cache['blocks.0.mlp.hook_pre'][:, -1]

        def compute_grams(cache):
            grams = {}
            which_layers = ["blocks.0.hook_resid_pre", "blocks.0.attn.hook_z", "blocks.0.hook_resid_mid",
                            "blocks.0.mlp.hook_post_hidden", "blocks.0.hook_resid_post", "hook_unembed_post"]
            act_dict = {k: v for (k,v) in cache.items() if k in which_layers}
            act_dict["blocks.0.attn.hook_z"] = torch.cat((act_dict["blocks.0.attn.hook_z"], act_dict["blocks.0.hook_resid_pre"]), dim=2)
            act_dict["blocks.0.mlp.hook_post_hidden"] = torch.cat((act_dict["blocks.0.mlp.hook_post_hidden"], act_dict["blocks.0.hook_resid_mid"]), dim=2)
            for name, activation in act_dict.items():
                # activation = activation[:,0,:]
                # activation = torch.tensor([
                #     [[1,2], [3,4], [5,6]],
                #     [[7,8], [9,10], [11,12]],
                #     [[13,14], [15,16], [17,18]],
                #     [[19,20], [21,22], [23,24]],
                # ])
                chunks = torch.split(activation, 100, dim=0)
                for chunk in chunks:
                    if name in grams:
                        grams[name] += torch.sum(torch.matmul(chunk.transpose(1, 2), chunk), dim=0)
                    else:
                        grams[name] = torch.sum(torch.matmul(chunk.transpose(1, 2), chunk), dim=0)
                        torch.cuda.empty_cache()
            grams = {k: v / len(activation) for k, v in grams.items()}
            return grams

        grams = compute_grams(cache)

        def compute_eigens(grams):
            eigenvalues, eigenvectors = {}, {}
            for name, gram in grams.items():
                lam, U = torch.linalg.eig(gram)
                eigenvalues[name] = torch.abs(lam)
                eigenvectors[name] = torch.abs(U)
            return eigenvalues, eigenvectors

        eigenvalues, eigenvectors = compute_eigens(grams)

        # Preprocess Eigens  TODO not sure how to handle this part yet
        # repeated_tensors = [lams[i].repeat(N[i]) for i in range(len(lams))]
        all_eigenvalues = torch.cat([lam.to("cpu").detach() for lam in eigenvalues.values()], dim=0).to("cpu").detach()

        scores.append(score)
        eigs.append(all_eigenvalues)

    # Plots Eigens
    n_bins = [100, 1000]
    save_path = "delete/"
    interactive_histogram(eigs, scores, which_models, n_bins, save_path)


def load_models(model_dir, device, which_models):
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

    models = []
    for m in range(len(which_models)):
        path = f"{model_dir}/{which_models[m]}.pth"
        run_saved_data = torch.load(path)
        model = Transformer(num_layers=num_layers, d_vocab=d_vocab, d_model=d_model, d_mlp=d_mlp, d_head=d_head,
                            num_heads=num_heads, n_ctx=n_ctx, act_type=act_type, use_cache=False, use_ln=use_ln)
        model.to(device)
        model.load_state_dict(run_saved_data['model'])
        models.append(model)
    return models


def cross_entropy_high_precision(logits, labels):
    # Shapes: batch x vocab, batch
    # Cast logits to float64 because log_softmax has a float32 underflow on overly
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes
    # and dodgy gradients
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss


if __name__ == "__main__":
    path = r"C:\Users\avery\Projects\alignment\ModularitySERI\saved_models\grokking\grok_1674663919"
    these_models = list(range(0,1000,100)) + list(range(1000,20000,1000))
    p = 113
    device = "cuda"

    fn_name = 'add'
    random_answers = np.random.randint(low=0, high=p, size=(p, p))
    fns_dict = {'add': lambda x, y: (x + y) % p, 'subtract': lambda x, y: (x - y) % p,
                'x2xyy2': lambda x, y: (x ** 2 + x * y + y ** 2) % p, 'rand': lambda x, y: random_answers[x][y]}
    fn = fns_dict[fn_name]
    all_data = torch.tensor([(i, j, p) for i in range(p) for j in range(p)]).to('cuda')
    all_labels = torch.tensor([fn(i, j) for i, j, _ in all_data]).to('cuda')

    models = load_models(path, device, these_models)
    main(models, all_data, all_labels, these_models)




