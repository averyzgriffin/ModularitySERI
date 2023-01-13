import torch


def compute_eigens(matrices: list):
    eigenvalues, eigenvectors = [],[]
    for N in matrices:
        evs, eigs = torch.linalg.eigh(N)
        eigenvalues.append(evs)
        eigenvectors.append(eigs)
    return eigenvectors, eigenvalues


def orthogonalize(model, U):
    ortho_layers = []
    for i in range(len(model.layers)):
        ortho_layers.append(torch.matmul(model.layers[i].weight, U[i].transpose(0,1)))
    return ortho_layers


def check_eigens(grams, U):
    for l in range(len(U)):
        check = torch.matmul(torch.matmul(U[l].transpose(0,1), grams[l]), U[l])
        print(f"Layer: {l} Is Diagonal: {is_diagonal(check)}")


def is_diagonal(A):
    # Create a matrix of the same shape as A with diagonal elements equal to the elements of A
    D = torch.diagflat(A.diag())
    # Check if the off-diagonal elements are zero
    off_diag = (D - A).abs().sum()
    # return off_diag == 0
    return off_diag