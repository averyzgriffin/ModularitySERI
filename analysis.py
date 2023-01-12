import matplotlib.pyplot as plt
import numpy as np
import torch


def preprocess_grams(lams: list, N: list):
    # Plot per layer
    # repeated_tensors = [lams[i].repeat(N[i]).detach() for i in range(len(lams))]
    # return repeated_tensors
    # Don't plot per layer
    repeated_tensors = [lams[i].repeat(N[i]) for i in range(len(lams))]
    return torch.cat(repeated_tensors, dim=0).detach()


def preprocess_lams_full_network(lams: list, N: list):
    x = [lams[0][N[i]:N[i+1]+1].repeat(N[i]) for i in range(len(N))]
    return torch.cat(x, dim=0).detach()


def repeat_and_concatenate(lam, N):
    ind = [9, 9, 5, 3]
    split_tensors = []
    start_idx = 0
    for i in ind:
        end_idx = start_idx + i
        split_tensors.append(lam[start_idx:end_idx])
        start_idx = end_idx
    repeated_tensors = [split_tensors[j].repeat(N[j]) for j in range(len(N))]
    concatenated_tensor = torch.cat(repeated_tensors)
    return concatenated_tensor.detach()


def plot_magnitude_frequency(values1, values2):
    # Compute the magnitudes of the values
    print("Min+Max eigenvalues of Gram Matrix: ", min(values1), max(values1))
    print("Min+Max eigenvalues of Hessian Matrix: ", min(values2), max(values2))
    magnitudes1 = torch.sqrt(torch.abs(values1*torch.sqrt(torch.tensor(2))))
    magnitudes2 = torch.sqrt(torch.abs(values2))

    # Count number of values in range
    count1 = torch.sum(torch.where(magnitudes1 < 0.5, 1, 0))
    count2 = torch.sum(torch.where(magnitudes2 < 0.25, 1, 0))
    print(f'Number of values in values1 that are less than 0.5: {count1}')
    print(f'Number of values in values2 that are less than 0.25: {count2}')

    # Determine the range of magnitudes
    min_mag = torch.min(torch.min(magnitudes1), torch.min(magnitudes2)).detach()
    max_mag = torch.max(torch.max(magnitudes1), torch.max(magnitudes2)).detach()

    # Create bins for the magnitudes
    bins = torch.linspace(min_mag, max_mag, steps=200)
    # x_tick_labels = [f'{a:.2f}-{b:.2f}' for a, b in zip(bins, bins[1:])]

    # Compute the histogram of magnitudes
    hist1, edges1 = torch.histogram(magnitudes1, bins=bins)
    hist2, edges2 = torch.histogram(magnitudes2, bins=bins)

    # Plot the histogram
    width = (edges1[1]-edges1[0]) / 2
    plt.bar(edges1[:-1], hist1, width=width,
            color='blue', label='Gram', alpha=.5)
    plt.bar(edges2[:-1]+width, hist2, width=width,
            color='orange', label='Hessian', alpha=.5)
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    # plt.xticks(edges1[:-1] + width/2, x_tick_labels, rotation=90)
    plt.legend()
    plt.show()


def plot_hessians(values1, values2):
    # Compute the magnitudes of the values
    print("Min+Max eigenvalues of Gram Matrix: ", min(values1), max(values1))
    print("Min+Max eigenvalues of Hessian Matrix: ", min(values2), max(values2))
    magnitudes1 = torch.sqrt(torch.abs(values1))
    magnitudes2 = torch.sqrt(torch.abs(values2))

    # Determine the range of magnitudes
    min_mag = torch.min(torch.min(magnitudes1), torch.min(magnitudes2)).detach()
    max_mag = torch.max(torch.max(magnitudes1), torch.max(magnitudes2)).detach()

    # Create bins for the magnitudes
    bins = torch.linspace(min_mag, max_mag, steps=200)
    # x_tick_labels = [f'{a:.2f}-{b:.2f}' for a, b in zip(bins, bins[1:])]

    # Compute the histogram of magnitudes
    hist1, edges1 = torch.histogram(magnitudes1, bins=bins)
    hist2, edges2 = torch.histogram(magnitudes2, bins=bins)

    # Plot the histogram
    width = (edges1[1]-edges1[0]) / 2
    plt.bar(edges1[:-1], hist1, width=width,
            color='blue', label='Hessian of less optimal network', alpha=.5)
    plt.bar(edges2[:-1]+width, hist2, width=width,
            color='orange', label='Hessian of more optimal network', alpha=.5)
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    # plt.xticks(edges1[:-1] + width/2, x_tick_labels, rotation=90)
    plt.legend()
    plt.show()


def plot_magnitude_frequency_by_layer(values1, values2):
    # Determine the range of magnitudes
    min_mag = min(min([torch.min(values1[i]) for i in range(len(values1))]), torch.min(torch.abs(values2)).detach()).detach()
    max_mag = max(max([torch.max(values1[i]) for i in range(len(values1))]), torch.max(torch.abs(values2)).detach()).detach()

    # Create bins for the magnitudes
    bins = torch.linspace(min_mag, max_mag, steps=100)

    fig, axes = plt.subplots(nrows=1, ncols=len(values1), figsize=(5 * len(values1), 5), sharex="all", sharey="all")

    for i, ax in enumerate(axes.flatten()):
        magnitudes1 = torch.sqrt(torch.abs(values1[i] * torch.sqrt(torch.tensor(2))))
        magnitudes2 = torch.sqrt(torch.abs(values2))

        # Compute the histograms of magnitudes
        hist1, edges1 = torch.histogram(magnitudes1, bins=bins)
        hist2, edges2 = torch.histogram(magnitudes2, bins=bins)

        width = (edges1[1] - edges1[0]) / 2
        # Plot the histograms
        ax.bar(edges1[:-1], hist1, width=width, color='blue', label='Gram-Norm')
        ax.bar(edges2[:-1] + width, hist2, width=width, color='orange', label='Hessian')
        ax.set_xlabel('Magnitude')
        ax.set_ylabel('Frequency')
        ax.legend()
    plt.show()






