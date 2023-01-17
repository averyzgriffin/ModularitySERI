import os
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from analysis import plot_all
from datasets import RetinaDataset
from eigen import compute_eigens
from gram import compute_grams, preprocess_lams, preprocess_lams_full_network, repeat_and_concatenate
from hessian import compute_hessian, manual_approximation
from models import OrthogMLP
from train import TrainerRetina, Trainer


conf_path = os.getcwd()
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def retina(batch_size, device):


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

    return [network1, network2], dataLoader


def mnist(batch_size, device, num_models, loss_fc, N):

    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', download=True, train=True,
                                                              transform=transforms.Compose([transforms.ToTensor(),
                                                                transforms.Normalize((0.1307,), (0.3081,))])),
                                                                batch_size=batch_size,
                                                                shuffle=True)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data',download=True,train=False,
                                                             transform=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Normalize((0.1307,), (0.3081,))])),
                                                            batch_size=1,
                                                            shuffle=True)

    models = []
    for m in range(num_models):
        print("Training Model # ", m)
        network = OrthogMLP(784, *N).to(device)
        epochs = 1

        trainer = Trainer(network, loss_fc, epochs, train_loader, device)
        trainer.train()
        models.append(network)

        # for test_image, test_label in test_loader:
        #     visualize_prediction(network, test_image, test_label)

    return models, train_loader


def visualize_prediction(model, test_image, test_label):
    model.eval()
    with torch.no_grad():
        prediction = model(test_image.reshape(1, -1))
        prediction = torch.argmax(prediction, dim=1)
        plt.imshow(test_image.squeeze(), cmap='gray')
        plt.title(f'Ground Truth: {test_label}  Prediction: {prediction.item()}')
        plt.show()


def compute_gram_hess_eigs(models: list, dataloader, loss_fc, N, per_layer, device):
    Gram_eigs = []
    Hess_eigs = []
    print("Computing Matrices")
    for network in models:

        # Gram Matrix
        if per_layer:
            grams = compute_grams(network, dataloader, per_layer=True)
            U, lam = compute_eigens(grams)
            lam = preprocess_lams(lam, N)
        else:
            grams = compute_grams(network, dataloader, per_layer=False)
            U, lam = compute_eigens(grams)

        # Hessian
        # h, h_u, h_lam = compute_hessian(network, dataloader, loss_fc, device)
        # Approximate Hessian
        # broadness = manual_approximation(network, loss_fc, dataloader, device)

        Gram_eigs.append(lam)
        # Hess_eigs.append(h_lam)
        # Hess_eigs.append(broadness)

    return Gram_eigs, Hess_eigs


def plot_eigens(gram_lams, hess_lams):
    print("Plots Eigens")
    # plot_magnitude_frequency(lam[0].detach(), h_lam)
    # plot_magnitude_frequency(lam, h_lam)
    # plot_hessians(h_lam, h_lam2)
    # plot_magnitude_frequency_by_layer(lam, h_lam)
    # plot_all(lam, lam_full, h_lam, lam2, lam_full2, h_lam2)
    pass


if __name__ == "__main__":
    batch_size = 1024
    device = torch.device("cpu")
    loss_fc = torch.nn.CrossEntropyLoss()
    N = [64, 10]
    num_models = 1
    per_layer = True

    # models = retina(batch_size)
    trained_models, dataloader = mnist(batch_size, device, num_models, loss_fc, N)
    gram_lams, hess_lams = compute_gram_hess_eigs(trained_models, dataloader, loss_fc, N, per_layer, device)
    plot_eigens(gram_lams, hess_lams)




