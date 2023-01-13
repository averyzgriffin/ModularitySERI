import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import _stateless

from analysis import plot_magnitude_frequency, plot_magnitude_frequency_by_layer, plot_hessians, plot_all
from datasets import RetinaDataset
from gram import compute_grams, preprocess_grams, preprocess_lams_full_network, repeat_and_concatenate
from hessian import compute_hessian
from models import OrthogMLP
from train import TrainerRetina


conf_path = os.getcwd()
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main(batch_size):

    device = torch.device("cpu")
    beltalowda = RetinaDataset(8)
    dataLoader = torch.utils.data.DataLoader(beltalowda, batch_size=batch_size)
    network1 = OrthogMLP(8, 8, 4, 2, 1).to(device)
    network2 = OrthogMLP(8, 8, 4, 2, 1).to(device)
    loss_fc = torch.nn.MSELoss()
    epochs1 = 200
    epochs2 = 5000
    goal_and = 1

    trainer1 = TrainerRetina(network1, loss_fc, epochs1, dataLoader, device)
    trainer2 = TrainerRetina(network2, loss_fc, epochs2, dataLoader, device)
    trainer1.train()
    trainer2.train()

    # Gram Matrix
    grams = compute_grams(network1, dataLoader, per_layer=True)
    grams2 = compute_grams(network2, dataLoader, per_layer=True)

    U, lam = compute_eigens(grams)
    U2, lam2 = compute_eigens(grams2)

    N = [8, 4, 2, 1]  # TODO hardcoded
    lam = preprocess_grams(lam, N)
    lam2 = preprocess_grams(lam2, N)

    grams_full = compute_grams(network1, dataLoader, per_layer=False)
    U_full, lam_full = compute_eigens(grams_full)
    lam_full = lam_full[0].detach()

    grams_full2 = compute_grams(network2, dataLoader, per_layer=False)
    U_full2, lam_full2 = compute_eigens(grams_full2)
    lam_full2 = lam_full2[0].detach()

    # Hessian
    print("Computing Hessian")
    h, h_u, h_lam = compute_hessian(network1, dataLoader, loss_fc, goal_and)
    h2, h_u2, h_lam2 = compute_hessian(network2, dataLoader, loss_fc, goal_and)

    # Analysis
    print("Plotting Frequencies")
    # plot_magnitude_frequency(lam[0].detach(), h_lam)
    # plot_magnitude_frequency(lam, h_lam)
    # plot_hessians(h_lam, h_lam2)
    # plot_magnitude_frequency_by_layer(lam, h_lam)
    plot_all(lam, lam_full, h_lam, lam2, lam_full2, h_lam2)


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


if __name__ == "__main__":
    batch_size = 256
    main(batch_size)



