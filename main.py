import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import _stateless

from analysis import plot_magnitude_frequency, plot_magnitude_frequency_by_layer, plot_hessians, plot_all
from datasets import RetinaDataset
from eigen import compute_eigens
from gram import compute_grams, preprocess_grams, preprocess_lams_full_network, repeat_and_concatenate
from hessian import compute_hessian
from models import OrthogMLP
from train import TrainerRetina


conf_path = os.getcwd()
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def retina(batch_size):

    device = torch.device("cpu")

    beltalowda = RetinaDataset(8)
    dataLoader = torch.utils.data.DataLoader(beltalowda, batch_size=batch_size)

    # Retina
    network1 = OrthogMLP(8, 8, 4, 2, 1).to(device)
    network2 = OrthogMLP(8, 8, 4, 2, 1).to(device)
    goal_and = 1

    loss_fc = torch.nn.MSELoss()
    epochs1 = 200
    epochs2 = 5000

    trainer1 = TrainerRetina(network1, loss_fc, epochs1, dataLoader, device)
    trainer2 = TrainerRetina(network2, loss_fc, epochs2, dataLoader, device)
    trainer1.train()
    trainer2.train()

    return [network1, network2]


def compute_matrices(models: list, dataloader, loss_fc, N):
    network1 = models[0]
    network2 = models[1]

    matrices = []
    # Gram Matrix
    grams = compute_grams(network1, dataloader, per_layer=True)
    grams2 = compute_grams(network2, dataloader, per_layer=True)

    U, lam = compute_eigens(grams)
    U2, lam2 = compute_eigens(grams2)

    # N = [8, 4, 2, 1]  # TODO hardcoded
    lam = preprocess_grams(lam, N)
    lam2 = preprocess_grams(lam2, N)

    grams_full = compute_grams(network1, dataloader, per_layer=False)
    U_full, lam_full = compute_eigens(grams_full)
    lam_full = lam_full[0].detach()

    grams_full2 = compute_grams(network2, dataloader, per_layer=False)
    U_full2, lam_full2 = compute_eigens(grams_full2)
    lam_full2 = lam_full2[0].detach()

    # Hessian
    print("Computing Hessian")
    goal_and = True  # TODO Hardcoded
    h, h_u, h_lam = compute_hessian(network1, dataloader, loss_fc, goal_and)
    h2, h_u2, h_lam2 = compute_hessian(network2, dataloader, loss_fc, goal_and)

    return matrices


def plot_eigens(eigens):
    # Analysis
    print("Plotting Frequencies")
    # plot_magnitude_frequency(lam[0].detach(), h_lam)
    # plot_magnitude_frequency(lam, h_lam)
    # plot_hessians(h_lam, h_lam2)
    # plot_magnitude_frequency_by_layer(lam, h_lam)
    # plot_all(lam, lam_full, h_lam, lam2, lam_full2, h_lam2)


if __name__ == "__main__":
    batch_size = 256
    retina(batch_size)



