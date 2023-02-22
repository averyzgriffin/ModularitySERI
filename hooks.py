import torch


def grab_activations(cache):
    which_layers = ["embed.hook_input", "blocks.0.hook_resid_pre", 'blocks.0.attn.hook_vprime_concat', 'blocks.0.hook_resid_mid'
                    ,'blocks.0.mlp.hook_hidden', 'blocks.0.hook_resid_post', 'hook_unembed_post']

    activations = {k: v for (k, v) in cache.items() if k in which_layers}

    batch_size = len(activations['blocks.0.hook_resid_mid'])
    seq_len = activations['blocks.0.hook_resid_mid'].shape[1]

    # Add bias to input, R2, and Hidden layer
    activations["embed.hook_input"] = torch.concat((torch.ones((batch_size,seq_len,1)).to("cuda:0"), activations['embed.hook_input']), dim=2)
    activations["blocks.0.hook_resid_mid"] = torch.concat((torch.ones((batch_size,seq_len,1)).to("cuda:0"), activations['blocks.0.hook_resid_mid']), dim=2)
    activations["blocks.0.mlp.hook_hidden"] = torch.concat((torch.ones((batch_size,seq_len,1)).to("cuda:0"), activations['blocks.0.mlp.hook_hidden']), dim=2)

    return activations


def grab_derivatives():
    pass


