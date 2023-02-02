import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from graphing import MNISTGraph, plot_modularity_plotly
from models import OrthogMLP
from train import Trainer



device = torch.device("cuda:0")
DATA_SIZE = 60000


def main(train, test, N, loss_fc, lr, opt, regularization, epochs):
    path = r"C:\Users\avery\Projects\alignment\ModularitySERI\saved_models\mnist\mnist_784x256x64x32x10_SGD_LR1_reg0\mnist_784x256x64x32x10_SGD_LR1_reg0_trial000\mnist_784x256x64x32x10_SGD_LR1_reg0_trial000_epoch099.pt"

    # network = build_model()
    network = load_model(path)

    gram = compute_gram(network, train)

    # Lam = eigenvalues, S = Inverse squareroot eigenvalues, U = eigenvectors
    lam, S, U = eigensolve(gram, object="gram")

    M = compute_M(network, gram, train)

    # D = eigenvalues, Dinvrs = Inverse squareroot eigenvalues, V = eigenvectors
    D, Dinvrs, V = eigensolve(M, object="M")

    new_edges = transform_network(network, train, U, S, V)

    # network = {1: torch.randn((256, 784)), 2: torch.randn((64, 256)),
    #            3: torch.randn((32, 64)), 4: torch.randn((10, 32))}
    # graph1 = MNISTGraph(network, absval=False)
    # Q, clusters = graph1.get_model_modularity(method="louvain")
    # plot_modularity_plotly(graph1, clusters)

    # each eigenvalue (D) divided by 2 is equal to the corresponding column sum in the edge matrix (row = to (L+1), column = from (L))
    # eigenvalues which are very negative should be flagged
    # rows of edges should sum to 1 or 0


def build_model():
    """ Build a model """
    model = OrthogMLP(4, 3, 2, 1)
    with torch.no_grad():
        model.fc0.weight = nn.Parameter(torch.ones_like(model.fc0.weight))
        model.fc1.weight = nn.Parameter(torch.tensor([[1., -2.], [3., -4.], [5., -6.]], requires_grad=True).T)
        model.fc2.weight = nn.Parameter(torch.tensor([[1., 1.]], requires_grad=True))
    return model


def train_model(train_loader, test_loader, N, loss_fc, lr, opt, regularization, epochs, save_path="", model_name=""):
    network = OrthogMLP(*N).to(device)
    trainer = Trainer(network, N, loss_fc, lr, opt, regularization, epochs, train_loader, test_loader, device, save_path, model_name)
    trainer.train()
    return network


def load_model(path):
    model = OrthogMLP(*N).to(device)
    model.load_state_dict(torch.load(path))
    return model


def compute_gram(model, dataloader):
    """ Compute the gram matrix for each layer of a network """

    # Adding hooks that will trigger when a sample x is passed through the model
    add_hooks(model, [model.grab_activations_hook])

    # Iterate through data
    for b, (x, label) in enumerate(dataloader):
        # Send x into model to trigger hooks
        model(x.reshape(len(x), -1).to(device))

        # Grab the activations that the hook stored
        activations = model.activations

        # If this is the first pass, instantiate an empty gram tensor in the dictionary to avoid key errors
        if b == 0:
            grams = {i: torch.zeros((act.shape[1], act.shape[1])).to(device) for i, act in enumerate(activations)}

        # Compute the gram matrix for each layer i, using the activations, for a single batch of data
        for i, act in enumerate(activations):

            # Gram matrix is defined as the inner product of the activations with its own transpose
            grams[i] += torch.matmul(act.transpose(0, 1), act)

    # Normalize each gram matrix by the number of samples |x|
    for k,v in grams.items():
        grams[k] = grams[k] / DATA_SIZE

    # Remove the hooks
    remove_hooks(model)
    return grams


def eigensolve(matrices: dict, object):
    """ Calculate the eigendata for a given matrix. """
    eigenvalues, norm_eigenvalues, eigenvectors = {}, {}, {}

    # Iterate through each matrix in matrices, one for each layer in the network
    for name, matrix in matrices.items():

        # Compute eigenvalues and eigenvectors
        lam, eigvec = torch.linalg.eig(matrix)

        # If this is the gram matrix, we take the absolute value of the eigenvalues
        if object == "gram":
            eigenvalues[name] = torch.abs(lam).real

        # If this is the M matrix, we don't take the absolute value
        elif object == "M":
            eigenvalues[name] = lam.real

        # Compute the inverse-squareroot of the eigenvalues. Each value is abs() first.
        # For any value that is close to 0, we just set it to zero to avoid infinities
        norm_eigenvalues[name] = torch.where(torch.abs(lam.real) >= 0.0001, torch.pow(torch.abs(lam), -0.5).real, torch.tensor([0.0], device=lam.device))

        # Eigenvectors are NOT abs()
        eigenvectors[name] = eigvec.real
    return eigenvalues, norm_eigenvalues, eigenvectors


def compute_M(model, grams, dataloader):
    M = {}

    # Add hooks that will trigger when a sample x is passed through the model
    add_hooks(model, hookfuncs=[model.compute_derivatives_hook])

    with torch.no_grad():
        # Iterate through the data
        for b, (x, label) in enumerate(dataloader):
            # Reset the derivatives before each pass
            model.derivatives = []
            print("M Computatation # samples processed ", b*len(x))

            # Pass a batch through the model to trigger hooks
            pred = model(x.reshape(len(x), -1).to(device))

            # Remove derivative connections to the nonexistent bias in the last layer (this only works for batches)
            # dfs should be: (n+1)x(m+1) each layer but (n+1)x(m) last layer; n := size of L, m:= L+1;
            model.derivatives[-1] = model.derivatives[-1][:,1:,:]

            # Compute the M matrix for each set of derivatives; corresponds to each layer except last one
            for n, df in enumerate(model.derivatives):
                # Grab the gram matrix for current layer
                gram = grams[n]

                # Set M value in dictionary to 0 to avoid key error
                M.setdefault(n, 0)

                # Compute M
                # Equivalent to M[n] += df.T @ df @ gram @ gram.T + gram @ gram.T @ df.T @ df
                M[n] += torch.sum( (torch.matmul(df.permute(0, 2, 1), torch.matmul(df, gram))) +
                                   (torch.matmul(gram, torch.matmul(df.permute(0, 2, 1), df))), dim=0)

            del x, label, model.derivatives, pred, n, df, gram

    # Normalize each M matrix by the number of samples |x|
    for k,v in M.items():
        M[k] = M[k] / DATA_SIZE

    # Remove the hooks
    remove_hooks(model)
    return M


def transform_network(model, dataloader, u, s, v):
    """ Calculate the new_edges of the model using the transformation matrices"""
    edges = {}

    # Add hooks that will trigger when a sample x is passed through the model
    add_hooks(model, [model.compute_derivatives_hook, model.grab_activations_hook])

    with torch.no_grad():
        # Iterate through the data
        for b, (x, label) in enumerate(dataloader):
            print("Edge Comp # samples processed ", b*len(x))

            # Reset the derivatives before each pass
            model.derivatives = []

            # Pass a batch through the model to trigger hooks
            pred = model(x.reshape(len(x), -1).to(device))

            # Remove derivative connections to the nonexistent bias in the last layer
            model.derivatives[-1] = model.derivatives[-1][:,1:,:]

            # Compute the new edges for each layer in the network. Do not include last layer in loop since that
            # computation is slightly different (doesn't include v)
            for l in range(len(model.layers)-1):

                # Convert the vector of eigenvalues s to a diagonal matrix, so we can do matrix multiplication
                s_d = torch.diag(s[l])

                # Compute the inverse of s. Where values = 0, we set value to 1 to avoid divide by zeros.
                s_invrse = torch.where(s_d != 0, torch.pow(s_d, -1), torch.tensor([1.0], device=s[l].device))

                # Compute the transformation of the L+1
                ls = torch.matmul(v[l + 1], torch.matmul(torch.diag(s[l + 1]), torch.matmul(u[l + 1], model.activations[l+1].T)))

                # Compute the transformation of the derivatives of L+1 to L
                m = torch.matmul(v[l+1], torch.matmul(torch.diag(s[l+1]), torch.matmul(u[l+1], torch.matmul(model.derivatives[l], torch.matmul(u[l].T, torch.matmul(s_invrse, v[l].T))))))

                # Compute the transformation of the L
                rs = torch.matmul(v[l], torch.matmul(torch.diag(s[l]), torch.matmul(u[l], model.activations[l].T)))

                # Set edges value in dictionary to 0 to avoid key error
                edges.setdefault(l, 0)

                value = torch.zeros_like(m[0])

                # For-loop for computing the edges for a particular layer and (summed across) a particular batch
                # Iterate through each sample in the batch, one at a time
                for i in range(len(m)):
                    # Reshape ls, m, and rs to be of the right shape after indexing into a specific sample of the batch
                    a = ls.T[i].reshape(1, ls.shape[0]).T
                    b = m[i]
                    c = rs.T[i].reshape(1, rs.shape[0])

                    # Compute the resulting value by scalar multiplying ls, m, and rs together. Add to value
                    value += a * b * c

                # Add the summed-across-each-sample-in-batch value to the dictionary
                edges[l] += value

                del value, a, b, c, ls, m, rs, s_d, s_invrse

            # Repeat all computations for the last layer. Only difference is that v is excluded.
            l += 1
            s_d = torch.diag(s[l])
            s_invrse = torch.where(s_d != 0, torch.pow(s_d, -1), torch.tensor([1.0], device=s[l].device))

            ls = torch.matmul(torch.diag(s[l + 1]), torch.matmul(u[l + 1], model.activations[l + 1].T))
            m = torch.matmul(torch.diag(s[l + 1]), torch.matmul(u[l + 1],torch.matmul(model.derivatives[l],torch.matmul(u[l].T, s_invrse))))
            rs = torch.matmul(torch.diag(s[l]), torch.matmul(u[l], model.activations[l].T))

            edges.setdefault(l, 0)
            value = torch.zeros_like(m[0])
            for i in range(len(ls)):
                a = ls.T[i].reshape(1, ls.shape[0]).T
                b = m[i]
                c = rs.T[i].reshape(1, rs.shape[0])
                value += a * b * c
            edges[l] += value

            del value, a, b, c, ls, m, rs, s_d, s_invrse, x, label, pred

    # Normalize each edge value by the number of samples |x|
    for k,v in edges.items():
        edges[k] = edges[k] / DATA_SIZE

    # Remove the hooks
    remove_hooks(model)
    return edges


def add_hooks(model, hookfuncs: list):
    model.handles = []
    for module in model.layers:
        for hook in hookfuncs:
            handle = module.register_forward_hook(hook)
            model.handles.append(handle)


def remove_hooks(model):
    for h in model.handles:
        h.remove()


if __name__ == '__main__':
    x = torch.ones((1, 1, 4), requires_grad=True)
    y = torch.tensor([42])
    data = [(x,y)]
    batch_size = 400
    loss_fc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD
    lr = .1
    N = [784, 256, 64, 32, 10]
    epochs = 100
    regularization = 0

    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', download=True, train=True,
                                                              transform=transforms.Compose([transforms.ToTensor(),
                                                                                            transforms.Normalize(
                                                                                                (0.1307,),
                                                                                                (0.3081,))])),
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', download=True, train=False,
                                                             transform=transforms.Compose([transforms.ToTensor(),
                                                                                           transforms.Normalize(
                                                                                               (0.1307,), (0.3081,))])),
                                              batch_size=1,
                                              shuffle=True)

    main(train_loader, test_loader, N, loss_fc, lr, optimizer, regularization, epochs)






