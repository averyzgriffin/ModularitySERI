import torch


device = torch.device("cuda:0")


def diag_input(model, u, s, s_invrs):
    WE = model.embed.W_E.detach()

    # Account for bias
    WP = model.pos_embed.W_pos.detach()
    max_values, _ = torch.max(WP, dim=0)
    max_values = max_values.view(-1, 1)
    WE = torch.cat([max_values, WE], dim=1)

    WE = torch.diag(s['blocks.0.hook_resid_pre']) @ u['blocks.0.hook_resid_pre'] @ WE @ u['embed.hook_input'].T @ torch.diag(s_invrs['embed.hook_input'])
    return 3 * torch.matmul(WE.T, WE)


def diag_residual_start(model, cache, u, s, s_invrs):
    sig = cache['blocks.0.attn.hook_post_softmax'][:,:,2,:]  # shape (50, 4, 3)
    WV = model.blocks[0].attn.W_V.detach()  # shape (4, 32, 128)

    sig = sig.unsqueeze(2)  # shape: (50, 4, 1, 3)

    # Transformation. shape (128, 128)
    WV = torch.diag(s['blocks.0.attn.hook_vprime_concat']) @ u['blocks.0.attn.hook_vprime_concat'] @  WV.view(128,128) @ u['blocks.0.hook_resid_pre'].T @ torch.diag(s_invrs['blocks.0.hook_resid_pre'])

    # Reshape to (4,32,3,128)
    WV = WV.view(4,32,128).unsqueeze(2).expand(4, 32, 3, 128)

    sig_wv = torch.einsum("bhet,hftd->bhftd", sig, WV)  # shape (50, 4, 32, 3, 128)
    temp = sig_wv.view(len(sig_wv), 128, 3, 128)
    temp1 = temp[:,:,0,:]
    temp2 = temp[:,:,1,:]
    temp3 = temp[:,:,2,:]

    squared_sig_wv1 = temp1.permute(0,2,1) @ temp1
    squared_sig_wv2 = temp2.permute(0,2,1) @ temp2
    squared_sig_wv3 = temp3.permute(0,2,1) @ temp3
    squared_sig_wv = squared_sig_wv1 + squared_sig_wv2 + squared_sig_wv3
    squared_sig_wv = squared_sig_wv.sum(0)

    I = torch.cat((torch.zeros((1, 128)), torch.eye(128)), dim=0).to(device)
    I = torch.diag(s['blocks.0.hook_resid_mid']) @ u['blocks.0.hook_resid_mid'] @ I @ u['blocks.0.hook_resid_pre'].T @ torch.diag(s_invrs['blocks.0.hook_resid_pre'])
    squared_I = torch.matmul(I.T, I)

    WE = model.embed.W_E.detach()
    # Account for bias
    WP = model.pos_embed.W_pos.detach()
    max_values, _ = torch.max(WP, dim=0)
    max_values = max_values.view(-1, 1)
    WE = torch.cat([max_values, WE], dim=1)
    WE = torch.diag(s['blocks.0.hook_resid_pre']) @ u['blocks.0.hook_resid_pre'] @ WE @ u['embed.hook_input'].T @ torch.diag(s_invrs['embed.hook_input'])
    squared_WE = 3 * torch.matmul(WE, WE.T)

    sum = squared_sig_wv + squared_I + squared_WE
    return sum


def diag_attention_out(model, cache, u, s, s_invrs):
    WZ = model.blocks[0].post_attn.W_O.detach()
    WZ = torch.cat((torch.zeros((1, 128)).to(device), WZ), dim=0)
    WZ = torch.diag(s['blocks.0.hook_resid_mid']) @ u['blocks.0.hook_resid_mid'] @  WZ @ u['blocks.0.attn.hook_vprime_concat'].T @ torch.diag(s_invrs['blocks.0.attn.hook_vprime_concat'])
    squared_wz = torch.matmul(WZ.T, WZ)

    sig = cache['blocks.0.attn.hook_post_softmax'][:,:,2,:]
    sig = sig.unsqueeze(2)

    WV = model.blocks[0].attn.W_V.detach()
    WV = torch.diag(s['blocks.0.attn.hook_vprime_concat']) @ u['blocks.0.attn.hook_vprime_concat'] @  WV.view(128,128) @ u['blocks.0.hook_resid_pre'].T @ torch.diag(s_invrs['blocks.0.hook_resid_pre'])
    WV = WV.view(4,32,128).unsqueeze(2).expand(4, 32, 3, 128)

    sig_wv = torch.einsum("bhet,hftd->bhftd", sig, WV)  # shape (50, 4, 32, 3, 128)
    temp = sig_wv.view(len(sig_wv), 128, 3, 128)
    temp1 = temp[:, :, 0, :]
    temp2 = temp[:, :, 1, :]
    temp3 = temp[:, :, 2, :]

    squared_sig_wv1 = temp1.permute(0, 2, 1) @ temp1
    squared_sig_wv2 = temp2.permute(0, 2, 1) @ temp2
    squared_sig_wv3 = temp3.permute(0, 2, 1) @ temp3
    squared_sig_wv = squared_sig_wv1 + squared_sig_wv2 + squared_sig_wv3
    squared_sig_wv = squared_sig_wv.sum(0)

    sum = squared_wz + squared_sig_wv
    return sum


def diag_residual_mid(model, cache, u, s, s_invrs):
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
    M = torch.diag(s['blocks.0.mlp.hook_hidden']) @ u['blocks.0.mlp.hook_hidden'] @  M @ u['blocks.0.hook_resid_mid'].T @ torch.diag(s_invrs['blocks.0.hook_resid_mid'])
    squared_M = torch.matmul(M.permute(0,2,1), M)

    II = torch.cat((torch.zeros((128, 1)), torch.eye(128)), dim=1).to(device)
    II = torch.diag(s['blocks.0.hook_resid_post']) @ u['blocks.0.hook_resid_post'] @ II @ u['blocks.0.hook_resid_mid'].T @ torch.diag(s_invrs['blocks.0.hook_resid_mid'])
    squared_II = torch.matmul(II.T, II)

    WZ = model.blocks[0].post_attn.W_O.detach()
    WZ = torch.cat((torch.zeros((1, 128)).to(device), WZ), dim=0)
    WZ = torch.diag(s['blocks.0.hook_resid_mid']) @ u['blocks.0.hook_resid_mid'] @  WZ @ u['blocks.0.attn.hook_vprime_concat'].T @ torch.diag(s_invrs['blocks.0.attn.hook_vprime_concat'])
    squared_wz = torch.matmul(WZ, WZ.T)

    I = torch.cat((torch.zeros((1, 128)), torch.eye(128)), dim=0).to(device)
    I = torch.diag(s['blocks.0.hook_resid_mid']) @ u['blocks.0.hook_resid_mid'] @ I @ u['blocks.0.hook_resid_pre'].T @ torch.diag(s_invrs['blocks.0.hook_resid_pre'])
    squared_I = torch.matmul(I, I.T)

    sum = squared_M + squared_II + squared_wz + squared_I
    return sum


def diag_hidden(model, cache, u, s, s_invrs):
    activations = cache['blocks.0.mlp.hook_hidden'][:, 2, :]
    activations = torch.cat((torch.ones((len(activations), 1)).to(device), activations), dim=1)
    preactivations = cache['blocks.0.mlp.hook_preact'][:, 2, :]
    preactivations = torch.cat((torch.ones((len(preactivations), 1)).to(device), preactivations), dim=1)
    ratio = torch.einsum('bi,bj->bij', activations, 1 / preactivations)
    eye = torch.eye(activations.shape[1], device=device).unsqueeze(0)
    ratio = ratio * eye  # Multiply identity matrix with diagonal elements

    WH = model.blocks[0].mlp.W_in.detach()
    b_in = model.blocks[0].mlp.b_in.detach()
    WH = torch.cat((b_in.unsqueeze(1), WH), dim=1)
    b_out = torch.zeros((WH.shape[1])).to(device)
    b_out[0] = 1
    WH = torch.cat((b_out.unsqueeze(0), WH), dim=0)

    M = torch.matmul(ratio, WH)
    M = torch.diag(s['blocks.0.mlp.hook_hidden']) @ u['blocks.0.mlp.hook_hidden'] @  M @ u['blocks.0.hook_resid_mid'].T @ torch.diag(s_invrs['blocks.0.hook_resid_mid'])
    squared_M = torch.matmul(M, M.permute(0,2,1))

    WO = model.blocks[0].mlp.W_out.detach()
    b_out = model.blocks[0].mlp.b_out.detach()
    WO = torch.cat((b_out.unsqueeze(1), WO), dim=1)
    WO = torch.diag(s['blocks.0.hook_resid_post']) @ u['blocks.0.hook_resid_post'] @  WO @ u['blocks.0.mlp.hook_hidden'].T @ torch.diag(s_invrs['blocks.0.mlp.hook_hidden'])
    squared_WO = torch.matmul(WO.T, WO)

    sum = squared_M + squared_WO
    return sum


def diag_output(model, u, s, s_invrs):
    WO = model.blocks[0].mlp.W_out.detach()
    b_out = model.blocks[0].mlp.b_out.detach()
    WO = torch.cat((b_out.unsqueeze(1), WO), dim=1)
    WO = torch.diag(s['blocks.0.hook_resid_post']) @ u['blocks.0.hook_resid_post'] @  WO @ u['blocks.0.mlp.hook_hidden'].T @ torch.diag(s_invrs['blocks.0.mlp.hook_hidden'])
    squared_WO = torch.matmul(WO, WO.T)

    II = torch.cat((torch.zeros((128, 1)), torch.eye(128)), dim=1).to(device)
    II = torch.diag(s['blocks.0.hook_resid_post']) @ u['blocks.0.hook_resid_post'] @ II @ u['blocks.0.hook_resid_mid'].T @ torch.diag(s_invrs['blocks.0.hook_resid_mid'])
    squared_II = torch.matmul(II, II.T)

    WP = model.unembed.W_U.detach().T
    WP = torch.diag(s['hook_unembed_post']) @ u['hook_unembed_post'] @ WP @ u['blocks.0.hook_resid_post'].T @ torch.diag(s_invrs['blocks.0.hook_resid_post'])
    squared_WP = torch.matmul(WP.T, WP)

    sum = squared_WO + squared_II + squared_WP
    return sum


def diag_unembed(model, u, s, s_invrs):
    WP = model.unembed.W_U.detach().T
    WP = torch.diag(s['hook_unembed_post']) @ u['hook_unembed_post'] @ WP @ u['blocks.0.hook_resid_post'].T @ torch.diag(s_invrs['blocks.0.hook_resid_post'])
    squared_WP = torch.matmul(WP, WP.T)
    return squared_WP






