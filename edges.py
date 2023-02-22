import torch


device = torch.device("cuda:0")


def edge_input(model, u, s, s_invrs, q):
    WE = model.embed.W_E.detach()

    # Account for bias
    WP = model.pos_embed.W_pos.detach()
    max_values, _ = torch.max(WP, dim=0)
    max_values = max_values.view(-1, 1)
    WE = torch.cat([max_values, WE], dim=1)

    WE = q['blocks.0.hook_resid_pre'] @ torch.diag(s['blocks.0.hook_resid_pre']) @ u['blocks.0.hook_resid_pre'] @ WE @ u['embed.hook_input'].T @ torch.diag(s_invrs['embed.hook_input']) @ q['embed.hook_input'].T

    return 3 * torch.square(WE)


def edge_residual12(model, cache, u, s, s_invrs, q):
    I = torch.cat((torch.zeros((1, 128)), torch.eye(128)), dim=0).to(device)
    I = q['blocks.0.hook_resid_mid'] @ torch.diag(s['blocks.0.hook_resid_mid']) @ u['blocks.0.hook_resid_mid'] @ I @ u['blocks.0.hook_resid_pre'].T @ torch.diag(s_invrs['blocks.0.hook_resid_pre']) @ q['blocks.0.hook_resid_pre'].T
    squared_I = torch.square(I)
    return squared_I


def edge_r1_attention(model, cache, u, s, s_invrs, q):

    # Grab the output of the attention filter, index the third token
    sig = cache['blocks.0.attn.hook_post_softmax'][:,:,2,:]  # shape is (50, 4, 3)

    # WV weight matrix
    WV = model.blocks[0].attn.W_V.detach()  # shape is (4, 32, 128)

    # Just adding an empty dimension for tensor multiplication purposes
    sig = sig.unsqueeze(2)  # shape: (50, 4, 1, 3)

    # Transformation. produces a tensor of shape (128, 128)
    WV = q['blocks.0.attn.hook_vprime_concat'] @ torch.diag(s['blocks.0.attn.hook_vprime_concat']) @ u['blocks.0.attn.hook_vprime_concat'] \
         @  WV.view(128,128) @ u['blocks.0.hook_resid_pre'].T @ torch.diag(s_invrs['blocks.0.hook_resid_pre']) @ q['blocks.0.hook_resid_pre']

    # Reshape WV to (4,32,3,128). (4,32) come from the first 128. 3 comes from having a (1,128) tensor for each token
    WV = WV.view(4,32,128).unsqueeze(2).expand(4, 32, 3, 128)

    # This step is doing the actual tensor multiplication
    sig_wv = torch.einsum("bhet,hftd->bhftd", sig, WV)  # shape (50, 4, 32, 3, 128)

    # Sum over the heads
    # sig_wv = sig_wv.sum(1).sum(1)  # TODO check that this is doing what its supposed to

    # Square
    squared_sig_wv = torch.square(sig_wv.view(len(sig_wv), 128, 3, 128))

    # Sum over the tokens
    squared_sig_wv = squared_sig_wv.sum(2)  # shape (50, 128)

    return squared_sig_wv


def edge_attention_r2(model, cache, u, s, s_invrs, q):
    WZ = model.blocks[0].post_attn.W_O.detach()
    WZ = torch.cat((torch.zeros((1, 128)).to(device), WZ), dim=0)
    WZ = q['blocks.0.hook_resid_mid'] @ torch.diag(s['blocks.0.hook_resid_mid']) @ u['blocks.0.hook_resid_mid'] @  WZ @ u['blocks.0.attn.hook_vprime_concat'].T @ torch.diag(s_invrs['blocks.0.attn.hook_vprime_concat']) @ q['blocks.0.attn.hook_vprime_concat'].T
    squared_wz = torch.square(WZ)
    return squared_wz


def edge_residual23(model, cache, u, s, s_invrs, q):
    II = torch.cat((torch.zeros((128, 1)), torch.eye(128)), dim=1).to(device)
    II = q['blocks.0.hook_resid_post'] @ torch.diag(s['blocks.0.hook_resid_post']) @ u['blocks.0.hook_resid_post'] @ II @ u[
        'blocks.0.hook_resid_mid'].T @ torch.diag(s_invrs['blocks.0.hook_resid_mid']) @ q['blocks.0.hook_resid_mid'].T
    squared_II = torch.square(II)
    return squared_II


def edge_r2_hidden(model, cache, u, s, s_invrs, q):
    # Activations / Preactivations
    activations = cache['blocks.0.mlp.hook_hidden'][:, 2, :]
    activations = torch.cat((torch.ones((len(activations), 1)).to(device), activations), dim=1)
    preactivations = cache['blocks.0.mlp.hook_preact'][:, 2, :]
    preactivations = torch.cat((torch.ones((len(preactivations), 1)).to(device), preactivations), dim=1)
    ratio = torch.einsum('bi,bj->bij', activations, 1 / preactivations)
    eye = torch.eye(activations.shape[1], device=device).unsqueeze(0)
    ratio = ratio * eye  # Multiply identity matrix with diagonal elements

    # WH
    WH = model.blocks[0].mlp.W_in.detach()
    b_in = model.blocks[0].mlp.b_in.detach()
    WH = torch.cat((b_in.unsqueeze(1), WH), dim=1)
    b_out = torch.zeros((WH.shape[1])).to(device)
    b_out[0] = 1
    WH = torch.cat((b_out.unsqueeze(0), WH), dim=0)

    # M(x)
    M = torch.matmul(ratio, WH)
    M = q['blocks.0.mlp.hook_hidden'] @ torch.diag(s['blocks.0.mlp.hook_hidden']) @ u['blocks.0.mlp.hook_hidden'] @  M @ u['blocks.0.hook_resid_mid'].T @ torch.diag(s_invrs['blocks.0.hook_resid_mid']) @ q['blocks.0.hook_resid_mid'].T
    squared_M = torch.square(M)
    return squared_M


def edge_hidden_r3(model, cache, u, s, s_invrs, q):
    WO = model.blocks[0].mlp.W_out.detach()
    b_out = model.blocks[0].mlp.b_out.detach()
    WO = torch.cat((b_out.unsqueeze(1), WO), dim=1)
    WO = q['blocks.0.hook_resid_post'] @ torch.diag(s['blocks.0.hook_resid_post']) @ u['blocks.0.hook_resid_post'] @  WO @ u['blocks.0.mlp.hook_hidden'].T @ torch.diag(s_invrs['blocks.0.mlp.hook_hidden']) @ q['blocks.0.mlp.hook_hidden'].T
    squared_WO = torch.square(WO)
    return squared_WO


def edge_R3_unembed(model, u, s, s_invrs, q):
    WP = model.unembed.W_U.detach().T
    WP = q['hook_unembed_post'] @ torch.diag(s['hook_unembed_post']) @ u['hook_unembed_post'] @ WP @ u['blocks.0.hook_resid_post'].T @ torch.diag(s_invrs['blocks.0.hook_resid_post']) @ q['blocks.0.hook_resid_post'].T
    squared_WP = torch.square(WP)
    return squared_WP





