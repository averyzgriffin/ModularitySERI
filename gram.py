import torch


class OrthogModel:

    def __init__(self, model, dataloader, loss_func):

        self.model = model
        self.dataloader = dataloader
        self.loss_func = loss_func


class OrthogW:

    def __init__(self, model, dataloader, loss_func):

        self.model = model
        self.dataloader = dataloader
        self.loss_func = loss_func


# Orthogonalize a particular weight matrix using UT - the matrix of eigenvectors
def orthogonalize_weights(weights, UT):
    return torch.matmul(weights, UT)


def orthognalize_functions(functions, U):
    return torch.matmal(functions, U)


def compute_hessian(model, calculate_loss_function):
    H = torch.autograd.functional.hessian(calculate_loss_function, tuple(model.parameters()))
    rows = []
    shapes = [p.shape for p in model.parameters()]
    for i in range(len(H)):
        rows.append(torch.cat([H[j][i].view(shapes[j].numel(), shapes[i].numel()) for j in range(len(H))], dim=0))

    full_hessian = torch.cat(rows, dim=1)
    return full_hessian


# Return U, N, UT
def compute_eigens(hessian):
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
    eigenvalues = eigenvalues.float()
    return eigenvectors, eigenvalues, eigenvectors.transpose(0,1)



