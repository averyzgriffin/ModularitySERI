import torch


def grab_activations(cache):
    which_layers = ["blocks.0.hook_resid_pre", 'blocks.0.post_attn.hook_z']

    activations = {k: v for (k, v) in cache.items() if k in which_layers}
    activations["blocks.0.post_attn.hook_z"] = torch.cat(
        (activations["blocks.0.post_attn.hook_z"], activations["blocks.0.hook_resid_pre"]), dim=2)
    # activations["blocks.0.mlp.hook_post_hidden"] = torch.cat(
    #     (activations["blocks.0.mlp.hook_post_hidden"], activations["blocks.0.hook_resid_mid"]), dim=2)

    return activations


def grab_derivatives():
    pass


