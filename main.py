import os
import torch
from torch.utils.data import DataLoader

from datasets import RetinaDataset
from models import OrthogMLP


conf_path = os.getcwd()
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main(batch_size):
    print(conf_path)
    device = torch.device("cpu")

    beltalowda = RetinaDataset(8)
    dataLoader = torch.utils.data.DataLoader(beltalowda, batch_size=batch_size)

    epochs = 1000
    goal_and = 1

    network = OrthogMLP(8, 8, 4, 2, 1).to(device)
    loss_fc = torch.nn.MSELoss()
    opt = torch.optim.Adam(lr=1e-3, params=network.parameters())

    for i in range(epochs):

        N,U = compute_eigens(network, dataLoader)
        # for l in range(len(U)):
        #     check = torch.matmul(torch.matmul(U[l].transpose(0,1), N[l]), U[l])
        #     print(f"Layer: {l} Is Diagonal: {is_diagonal(check)}")
        ortho_layers = orthogonalize(network, U)

        for b, (x, label) in enumerate(dataLoader):
            x = x.to(device)
            _and, _or = label
            result = _and if goal_and else _or
            prediction = network(x)
            opt.zero_grad()
            loss = loss_fc(prediction.view(-1), result.float().to(device))
            loss.backward()
            opt.step()


def compute_eigens(model, dataloader):
    Grams = []
    for b, (x, label) in enumerate(dataloader):
        prediction = model(x)
        activations = model.activations
        for layer in activations:
            gram = 0
            for sample in layer:
                functions = sample.reshape(sample.shape[0], 1)
                gram += torch.matmul(functions, functions.transpose(0,1))
            Grams.append(gram)

    eigenvalues, eigenvectors = [],[]
    for N in Grams:
        evs, eigs = torch.linalg.eigh(N)
        eigenvalues.append(evs)
        eigenvectors.append(eigs)
    return Grams, eigenvectors


def orthogonalize(model, U):
    ortho_layers = []
    for i in range(len(model.layers)):
        ortho_layers.append(torch.matmul(model.layers[i].weight, U[i].transpose(0,1)))
    return ortho_layers


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



