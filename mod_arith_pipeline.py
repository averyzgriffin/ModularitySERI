import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from analysis import interactive_histogram
from datasets import ModularArithmeticDataset
import diagonalization as diag
import edges as edge
from hooks import grab_activations
from models import OrthogMLP, Transformer
from train import Trainer



device = torch.device("cuda:0")
DATA_SIZE = 1234


def main(networks, train, test, which_models):
    all_edges = []
    test_scores = []
    eigs = []

    for network in networks:

        loss = get_loss(network, test)
        test_scores.append(loss)

        gram = compute_gram(network, train)

        keys = list(gram.keys())

        # Lam = eigenvalues, S = negative squareroot of lam, U = eigenvectors, Spseudo = pseudo inverse of S
        lam, S, U, Spseudo = eigensolve(gram, object="gram", threshold=0.0001)

        M = compute_diag(network, train, U, S, Spseudo, keys)

        # Eigendata for the M matrix
        D, Dnorm, Q, Dpsuedo = eigensolve(M, object="M", threshold=0.0001)

        # Transforming the edges to new basis
        new_edges = compute_edges(network, train, U, S, Spseudo, Q, keys)

        merged_edges = torch.cat([edge.flatten().to("cpu") for edge in new_edges.values()])
        all_edges.append(merged_edges)

        all_eigenvalues = torch.cat([lam.to("cpu").detach() for lam in lam.values()], dim=0).to("cpu").detach()
        eigs.append(all_eigenvalues)

    n_bins = [100, 1000, 10000]
    save_path = "del/"
    interactive_histogram(all_edges, test_scores, which_models, n_bins, save_path, name="Edges")
    interactive_histogram(eigs, test_scores, which_models, n_bins, save_path, name="Eigenvalues of Gram Matrices")


def get_loss(model, data):
    losses = []
    for b, (x, label) in enumerate(data):
        logits = model(x.reshape(len(x), -1).to(device))[:, -1]
        logits = logits[:, :-1]
        loss = cross_entropy_high_precision(logits, label)
        loss = loss.to("cpu").detach().float()
        losses.append(loss)
    return torch.mean(torch.tensor(losses))


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


def compute_diag(model, dataloader, u, s, s_invrs, keys):
    M = {}
    cache = {}
    model.remove_all_hooks()
    model.cache_all(cache)

    with torch.no_grad():
        # Iterate through the data
        for b, (x, label) in enumerate(dataloader):
            # if b == 5: break
            print("M batch", b)
            if b == 63:
                print()

            prediction = model(x)[:, -1]  # Call the model such that the forward hooks are triggered
            activations = grab_activations(cache)  # Grab the specific layers we want

            # Input layer
            DI = diag.diag_input(model, u, s, s_invrs)

            # First residual start
            DR1 = diag.diag_residual_start(model, cache, u, s, s_invrs)

            # Attention Output
            DV = diag.diag_attention_out(model, cache, u, s, s_invrs)

            # Attention residual mid
            DR2 = diag.diag_residual_mid(model, cache, u, s, s_invrs)

            # MLP Hidden Layer
            DH = diag.diag_hidden(model, cache, u, s, s_invrs)

            # MLP Output
            DO = diag.diag_output(model, u, s, s_invrs)

            # Unembedding
            DU = diag.diag_unembed(model, u, s, s_invrs)

            matrices = [DI, DR1, DV, DR2, DH, DO, DU]

            for n in range(len(matrices)):
                # Set M value in dictionary to 0 to avoid key error
                M.setdefault(keys[n], 0)
                if len(matrices[n].shape) == 3:
                    matrices[n] = torch.sum(matrices[n], dim=0)
                M[keys[n]] += matrices[n]

            del x, label, matrices, DI, DR1, DV, DR2, DH, DO, DU, prediction, activations

    # Normalize each M matrix by the number of samples |x|
    for k, v in M.items():
        M[k] = M[k] / len(dataloader.dataset)

    return M


def compute_edges(model, dataloader, u, s, s_invrs, q, keys):
    edges = {}
    cache = {}
    model.remove_all_hooks()
    model.cache_all(cache)

    with torch.no_grad():
        # Iterate through the data
        for b, (x, label) in enumerate(dataloader):
            print("Edge batch", b)

            prediction = model(x)[:, -1]  # Call the model such that the forward hooks are triggered
            activations = grab_activations(cache)  # Grab the specific layers we want

            # Input layer
            EI = edge.edge_input(model, u, s, s_invrs, q)

            # First residual1 to 2
            ER12 = edge.edge_residual12(model, cache, u, s, s_invrs, q)

            # First residual1 to attention output
            ER1V = edge.edge_r1_attention(model, cache, u, s, s_invrs, q)

            # Attention to R2
            EVR2 = edge.edge_attention_r2(model, cache, u, s, s_invrs, q)

            # R2 to R3
            ER23 = edge.edge_residual23(model, cache, u, s, s_invrs, q)

            # R2 to MLP Hidden Layer
            ER2H = edge.edge_r2_hidden(model, cache, u, s, s_invrs, q)

            # MLP Hidden to R3
            EHR3 = edge.edge_hidden_r3(model, cache, u, s, s_invrs, q)

            # R3 to Unembedding
            ER3U = edge.edge_R3_unembed(model, u, s, s_invrs, q)

            matrices = [EI, ER12, ER1V, EVR2, ER23, ER2H, EHR3, ER3U]

            for n in range(len(matrices)-1):
                # Set value in dictionary to 0 to avoid key error
                edges.setdefault(keys[n], 0)
                if len(matrices[n].shape) == 3:
                    matrices[n] = torch.sum(matrices[n], dim=0)
                edges[keys[n]] += matrices[n]

            del x, label, matrices, EI, ER12, ER1V, EVR2, ER23, ER2H, EHR3, ER3U, prediction, activations

    # Normalize each M matrix by the number of samples |x|
    for k, v in edges.items():
        edges[k] = edges[k] / len(dataloader.dataset)

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


def load_models(model_dir, device, which_models):
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

    models = []
    for m in range(len(which_models)):
        path = f"{model_dir}/{which_models[m]}.pth"
        run_saved_data = torch.load(path)
        model = Transformer(num_layers=num_layers, d_vocab=d_vocab, d_model=d_model, d_mlp=d_mlp, d_head=d_head,
                            num_heads=num_heads, n_ctx=n_ctx, act_type=act_type, use_cache=False, use_ln=use_ln)
        model.to(device)
        model.load_state_dict(run_saved_data['model'])
        models.append(model)
    return models


def cross_entropy_high_precision(logits, labels):
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss


if __name__ == '__main__':
    path = r"C:\Users\Avery\Projects\ModularitySERI\saved_models\modular_addition\02_22_23"
    these_models = list(range(0,10000,1000)) + list(range(10000,30000,5000))
    networks = load_models(path, device, these_models)

    p = 113
    fn_name = 'add'

    train_data = ModularArithmeticDataset(p, fn_name, device, split=.25, seed=0, train=True)
    test_data = ModularArithmeticDataset(p, fn_name, device, split=.25, seed=0, train=False)

    train_loader = DataLoader(train_data, batch_size=200, shuffle=False)  # do not shuffle since it is already shuffled
    test_loader = DataLoader(test_data, batch_size=200, shuffle=False)

    main(networks, train_loader, test_loader, these_models)





