import matplotlib.pyplot as plt
import numpy as np
import torch


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


def plot_all(eigs_layer, eigs_net, eigs_hess, eigs_layer2, eigs_net2, eigs_hess2):
    magnitudes1 = torch.sqrt(torch.abs(eigs_layer*torch.sqrt(torch.tensor(2))))  # TODO ask if we should be scaling still
    magnitudes2 = torch.sqrt(torch.abs(eigs_net*torch.sqrt(torch.tensor(2))))
    magnitudes3 = torch.sqrt(torch.abs(eigs_hess))

    magnitudes4 = torch.sqrt(torch.abs(eigs_layer2*torch.sqrt(torch.tensor(2))))  # TODO ask if we should be scaling still
    magnitudes5 = torch.sqrt(torch.abs(eigs_net2*torch.sqrt(torch.tensor(2))))
    magnitudes6 = torch.sqrt(torch.abs(eigs_hess2))

    # Count number of values in range
    count1 = torch.sum(torch.where(magnitudes1 < 0.4, 1, 0))
    # count2 = torch.sum(torch.where(magnitudes2 < 0.25, 1, 0))
    count3 = torch.sum(torch.where(magnitudes3 < 0.1, 1, 0))
    count4 = torch.sum(torch.where(magnitudes4 < 0.4, 1, 0))
    # count5 = torch.sum(torch.where(magnitudes5 < 0.25, 1, 0))
    count6 = torch.sum(torch.where(magnitudes6 < 0.1, 1, 0))

    print(f'Number of network 1 gram-layer eigenvalues less than 0.4: {count1}')
    print(f'Number of network 2 gram-layer eigenvalues less than 0.4: {count4}')
    print(f'Number of network 1 hessian eigenvalues less than 0.1: {count3}')
    print(f'Number of network 2 hessian eigenvalues less than 0.1: {count6}')

    # Determine the range of magnitudes
    min_mag = min(torch.min(magnitudes1), torch.min(magnitudes2), torch.min(magnitudes3),
                  torch.min(magnitudes4), torch.min(magnitudes5), torch.min(magnitudes6)).detach()
    max_mag = max(torch.max(magnitudes1), torch.max(magnitudes2), torch.max(magnitudes3),
                  torch.max(magnitudes4), torch.max(magnitudes5), torch.max(magnitudes6)).detach()

    # Create bins for the magnitudes
    bins = torch.linspace(min_mag, max_mag, steps=150)

    # Compute the histogram of magnitudes
    hist1, edges1 = torch.histogram(magnitudes1, bins=bins)
    hist2, edges2 = torch.histogram(magnitudes2, bins=bins)
    hist3, edges3 = torch.histogram(magnitudes3, bins=bins)
    hist4, edges4 = torch.histogram(magnitudes4, bins=bins)
    hist5, edges5 = torch.histogram(magnitudes5, bins=bins)
    hist6, edges6 = torch.histogram(magnitudes6, bins=bins)

    fig, axes = plt.subplots(nrows=3, ncols=2,  sharex="all", sharey="all")
    fig.suptitle('Eigenvalues')

    width1 = (edges1[1] - edges1[0]) * .75
    width2 = (edges2[1] - edges2[0]) * .75
    width3 = (edges3[1] - edges3[0]) * .75
    width4 = (edges4[1] - edges4[0]) * .75
    width5 = (edges5[1] - edges5[0]) * .75
    width6 = (edges6[1] - edges6[0]) * .75
    axes[0][0].bar(edges1[:-1], hist1, width=width1, color='blue', label='Gram Per Layer')
    axes[1][0].bar(edges2[:-1], hist2, width=width2, color='orange', label='Gram Full Network')
    axes[2][0].bar(edges3[:-1], hist3, width=width3, color='green', label='Hessian')
    axes[0][1].bar(edges4[:-1], hist4, width=width4, color='blue', label='Gram Per Layer')
    axes[1][1].bar(edges5[:-1], hist5, width=width5, color='orange', label='Gram Full Network')
    axes[2][1].bar(edges6[:-1], hist6, width=width6, color='green', label='Hessian')

    axes[1][0].set_ylabel('Frequency')
    axes[2][1].set_xlabel('Magnitude')
    axes[0][1].legend()
    axes[1][1].legend()
    axes[2][1].legend()

    plt.show()






