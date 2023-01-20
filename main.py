import os
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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


def mnist(batch_size, device, loss_fc, lr, opt, regularization, N, epochs, save_path, model_name):

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

    network = OrthogMLP(*N).to(device)
    trainer = Trainer(network, N, loss_fc, lr, opt, regularization, epochs, train_loader, test_loader, device, save_path, model_name)
    trainer.train()

    return network


def visualize_prediction(model, test_image, test_label):
    model.eval()
    with torch.no_grad():
        prediction = model(test_image.reshape(1, -1))
        prediction = torch.argmax(prediction, dim=1)
        plt.imshow(test_image.squeeze(), cmap='gray')
        plt.title(f'Ground Truth: {test_label}  Prediction: {prediction.item()}')
        plt.show()


def compute_gram_eigs(models: list, dataloader, N, per_layer, device):
    Gram_eigs = []

    print("Computing Grams")
    for network in models:
        if per_layer:
            grams = compute_grams(network, dataloader, True, device)
            U, lam = compute_eigens(grams)
            lam = preprocess_lams(lam, N)
        else:
            grams = compute_grams(network, dataloader, False, device)
            U, lam = compute_eigens(grams)

        Gram_eigs.append(lam)

    return Gram_eigs


def compute_hess_eigs(models: list, dataloader, loss_fc, device):
    Hess_eigs = []

    print("Computing Hessians")
    for network in models:
        h, h_u, h_lam = compute_hessian(network, dataloader, loss_fc, device)
        # h = manual_approximation(network, loss_fc, dataloader, device)
        Hess_eigs.append(h_lam)

    return Hess_eigs


def create_model_name(task, optimizer, lr, N, regularization):
    optimizer_name = optimizer.__name__
    lr_str = str(lr).replace(".", "")
    N_str = "x".join(str(n) for n in N)
    name = f"{task}_{N_str}_{optimizer_name}_LR{lr_str}_reg{regularization}"
    return name


if __name__ == "__main__":
    batch_size = 2048
    device = torch.device("cuda:0")
    loss_fc = torch.nn.CrossEntropyLoss()
    task = "mnist"
    optimizer = torch.optim.SGD
    lr = .01
    N = [784, 500, 10]
    epochs = 2
    num_models = 4
    regularization = 0
    # model_name_check = "mnist_512x256x64_SGD_LR1e1_reg0"
    # save_path = f"saved_models/512_256_64_SGD_{i}"

    for i in range(num_models):
        model_name = create_model_name(task, optimizer, lr, N, regularization)
        save_path = f"saved_models/{task}/{model_name}/{model_name}_trial{str(i).zfill(3)}"
        os.makedirs(save_path, exist_ok=True)
        model_name = model_name+f"_trial{str(i).zfill(3)}"

        trained_models = mnist(batch_size, device, loss_fc, lr, optimizer, regularization, N, epochs, save_path, model_name)
        # models = retina(batch_size)
        # gram_lams = compute_gram_eigs(trained_models, dataloader, N, per_layer, device)
        # hess_lams = compute_hess_eigs(trained_models, dataloader, loss_fc, device)

