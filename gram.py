import torch


def compute_grams(model, dataloader, per_layer=True):
    Grams = []
    for b, (x, label) in enumerate(dataloader):  # TODO we probably want a different dataLoader for this step
        prediction = model(x)
        activations = model.activations
        if per_layer:
            for layer in activations:
                gram = torch.matmul(layer.transpose(0, 1), layer)
                gram = gram / 256
                Grams.append(gram)
        else:
            f = torch.cat(activations, dim=1)
            gram = torch.matmul(f.transpose(0, 1), f) / 256
            Grams.append(gram)
    return Grams


def preprocess_grams(lams: list, N: list):
    # Plot per layer
    # repeated_tensors = [lams[i].repeat(N[i]).detach() for i in range(len(lams))]
    # return repeated_tensors
    # Don't plot per layer
    repeated_tensors = [lams[i].repeat(N[i]) for i in range(len(lams))]
    return torch.cat(repeated_tensors, dim=0).detach()


def preprocess_lams_full_network(lams: list, N: list):
    x = [lams[0][N[i]:N[i+1]+1].repeat(N[i]) for i in range(len(N))]
    return torch.cat(x, dim=0).detach()


def repeat_and_concatenate(lam, N):
    ind = [9, 9, 5, 3]
    split_tensors = []
    start_idx = 0
    for i in ind:
        end_idx = start_idx + i
        split_tensors.append(lam[start_idx:end_idx])
        start_idx = end_idx
    repeated_tensors = [split_tensors[j].repeat(N[j]) for j in range(len(N))]
    concatenated_tensor = torch.cat(repeated_tensors)
    return concatenated_tensor.detach()


# class OrthogModel:
#
#     def __init__(self, model, dataloader, loss_func):
#
#         self.model = model
#         self.dataloader = dataloader
#         self.loss_func = loss_func
#
#
# class OrthogW:
#
#     def __init__(self, model, dataloader, loss_func):
#
#         self.model = model
#         self.dataloader = dataloader
#         self.loss_func = loss_func
#
#
# # Orthogonalize a particular weight matrix using UT - the matrix of eigenvectors
# def orthogonalize_weights(weights, UT):
#     return torch.matmul(weights, UT)
#
#
# def orthognalize_functions(functions, U):
#     return torch.matmal(functions, U)
#
#
# def compute_hessian(model, calculate_loss_function):
#     H = torch.autograd.functional.hessian(calculate_loss_function, tuple(model.parameters()))
#     rows = []
#     shapes = [p.shape for p in model.parameters()]
#     for i in range(len(H)):
#         rows.append(torch.cat([H[j][i].view(shapes[j].numel(), shapes[i].numel()) for j in range(len(H))], dim=0))
#
#     full_hessian = torch.cat(rows, dim=1)
#     return full_hessian
#
#
# # Return U, N, UT
# def compute_eigens(hessian):
#     eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
#     eigenvalues = eigenvalues.float()
#     return eigenvectors, eigenvalues, eigenvectors.transpose(0,1)



