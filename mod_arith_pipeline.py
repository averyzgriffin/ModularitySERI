import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from datasets import ModularArithmeticDataset
from hooks import grab_activations
from models import OrthogMLP, Transformer
from train import Trainer


device = torch.device("cuda:0")
DATA_SIZE = 1234


def main(network, train):

    gram = compute_gram(network, train)

    # Lam = eigenvalues, S = negative squareroot of lam, U = eigenvectors, Spseudo = pseudo inverse of S
    lam, S, U, Spseudo = eigensolve(gram, object="gram", threshold=0.0001)

    M = compute_M(network, gram, train, U, S, Spseudo)

    # Eigendata for the M matrix
    D, Dnorm, V, Dpsuedo = eigensolve(M, object="M", threshold=0.0001)

    # Transforming the edges to new basis
    new_edges = transform_network2(network, train, U, S, Spseudo, V)


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


def load_models(model_dir, device):
    p = 113
    d_model = 128
    num_layers = 1
    d_vocab = p + 1
    n_ctx = 3
    d_mlp = 4 * d_model
    num_heads = 4
    assert d_model % num_heads == 0
    d_head = d_model // num_heads
    act_type = 'ReLU'
    use_ln = False

    run_saved_data = torch.load(model_dir)
    model = Transformer(num_layers=num_layers, d_vocab=d_vocab, d_model=d_model, d_mlp=d_mlp, d_head=d_head,
                        num_heads=num_heads, n_ctx=n_ctx, act_type=act_type, use_cache=False, use_ln=use_ln)
    model.to(device)
    model.load_state_dict(run_saved_data['model'])
    return model


def compute_gram(model, dataloader):
    """ Compute the gram matrix for each layer of a network """

    # Adding hooks that will trigger when a sample x is passed through the model
    # add_hooks(model, [model.grab_activations_hook]) # this is something I may try implementing later

    # Iterate through data
    for b, (x, label) in enumerate(dataloader):
        print("Gram batch # ", b)

        # Set the activation hooks for the transformer class
        cache = {}
        model.remove_all_hooks()
        model.cache_all(cache)
        # model.setup_hooks(Transformer.hook_activations)  # this is something I may try implementing later

        # Send x into model to trigger hooks
        model(x.reshape(len(x), -1).to(device))

        # Grab the activations that the hook stored
        activations = grab_activations(cache)

        # If this is the first pass, instantiate an empty gram tensor in the dictionary to avoid key errors
        if b == 0:
            grams = {i: torch.zeros((act.shape[2], act.shape[2])).to(device) for i, act in activations.items()}

        # Compute the gram matrix for each layer i, using the activations, for a single batch of data
        for i, act in activations.items():
            grams[i] += torch.sum(torch.matmul(act.transpose(1, 2), act), dim=0)

    # Normalize each gram matrix by the number of samples |x|
    for k,v in grams.items():
        grams[k] = grams[k] / len(dataloader.dataset)

    # Remove the hooks
    model.remove_all_hooks()
    return grams


def eigensolve(matrices: dict, object, threshold):
    """ Calculate the eigendata for a given matrix. """
    eigenvalues, s, eigenvectors, psuedo_inverse_s = {}, {}, {}, {}

    # Iterate through each matrix in matrices, one for each layer in the network
    for name, matrix in matrices.items():

        # Compute eigenvalues and eigenvectors (columns are the eigenvectors)
        lam, eigvec = torch.linalg.eig(matrix)

        # If this is the gram matrix, we take the absolute value of the eigenvalues
        if object == "gram":
            eigenvalues[name] = torch.abs(lam).real

        # If this is the M matrix, we don't take the absolute value
        elif object == "M":
            eigenvalues[name] = lam.real

        # Compute the inverse-squareroot of the eigenvalues. Each value is abs() first.
        # For any value that is close to 0, we just set it to zero to avoid infinities
        s[name] = torch.where(torch.abs(lam.real) >= threshold, torch.pow(torch.abs(lam), -0.5).real, torch.tensor([0.0], device=lam.device))

        # Words
        psuedo_inverse_s[name] = torch.where(torch.abs(lam.real) >= threshold, torch.pow(torch.abs(lam), 0.5).real, torch.tensor([0.0], device=lam.device))

        # Eigenvectors are NOT abs()
        eigenvectors[name] = eigvec.real
    return eigenvalues, s, eigenvectors, psuedo_inverse_s


def compute_M(model, grams, dataloader, u, s, s_invrs):
    M = {}
    func_derivatives = {}

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

                # Transform the functional derivatives
                transformed_df = transform_derivatives(df, n, u, s, s_invrs)

                # Grab the gram matrix for current layer
                gram = grams[n]

                # Set M value in dictionary to 0 to avoid key error
                M.setdefault(n, 0)

                # Compute M
                # Equivalent to M[n] += df.T @ df @ gram @ gram.T + gram @ gram.T @ df.T @ df
                M[n] += torch.sum( (torch.matmul(transformed_df.permute(0, 2, 1), torch.matmul(transformed_df, gram))) +
                                   (torch.matmul(gram, torch.matmul(transformed_df.permute(0, 2, 1), transformed_df))), dim=0)

            del x, label, model.derivatives, pred, n, df, gram

    # Normalize each M matrix by the number of samples |x|
    for k,v in M.items():
        M[k] = M[k] / len(dataloader.dataset)

    # Remove the hooks
    remove_hooks(model)
    return M


def transform_network2(model, dataloader, u, s, s_invrs, v):
    new_edges = {}

    # Add hooks that will trigger when a sample x is passed through the model
    add_hooks(model, hookfuncs=[model.compute_derivatives_hook])

    with torch.no_grad():
        # Iterate through the data
        for b, (x, label) in enumerate(dataloader):
            # Reset the derivatives before each pass
            model.derivatives = []
            print("New Edges Computatation # samples processed ", b * len(x))

            # Pass a batch through the model to trigger hooks
            pred = model(x.reshape(len(x), -1).to(device))

            # Remove derivative connections to the nonexistent bias in the last layer (this only works for batches)
            # dfs should be: (n+1)x(m+1) each layer but (n+1)x(m) last layer; n := size of L, m:= L+1;
            model.derivatives[-1] = model.derivatives[-1][:, 1:, :]

            for n in range(len(model.derivatives)-1):
                # Transform the functional derivatives
                df = model.derivatives[n]
                transformed_df = transform_derivatives(df, n, u, s, s_invrs)

                # Set new edges value in dictionary to 0 to avoid key error
                new_edges.setdefault(n, 0)

                # Compute new edges
                a = torch.matmul(v[n+1].T, torch.matmul(transformed_df, v[n]))
                new_edges[n] += torch.sum(torch.pow(a, 2), axis=0)

            del x, label, model.derivatives, pred, n, df, a, transformed_df

    # Normalize each M matrix by the number of samples |x|
    for k, v in new_edges.items():
        new_edges[k] = new_edges[k] / len(dataloader.dataset)

    # Remove the hooks
    remove_hooks(model)
    return new_edges


def transform_derivatives(df, n, u, s, s_invsr):
    new_ds = []
    for i,d in enumerate(df):
        new_d = torch.matmul(torch.diag(s[n+1]), torch.matmul(u[n+1].T, torch.matmul(d, torch.matmul(u[n], torch.diag(s_invsr[n])))))
        new_ds.append(new_d)
    new_ds = torch.stack(new_ds)
    return new_ds


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
    path = r"C:\Users\Avery\Projects\ModularitySERI\saved_models\default101_final.pth"
    model = load_model(path, device)

    p = 113
    fn_name = 'add'
    dataset = ModularArithmeticDataset(p, fn_name, device)

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    main(model, dataloader)





