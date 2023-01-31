import itertools
import networkx as nx
from networkx import edge_betweenness_centrality as betweenness
import torch.nn as nn

from sklearn.cluster import SpectralClustering


def peek(it):
    """
    Peek at the next item of an iterator

    This is done by requesting the next item from the iterator and immediately pushing it back

    :param it: the iterator
    :return: the next element, the reset iterator
    """
    first = next(it)
    return first, itertools.chain([first], it)


class ModelGraph(nx.Graph):
    """
    Build a graph from an MLP. This enables us to perform graph algorithms on it.

    One important usage is the calculation of the network modularity using the model weights as edge weights.
    """
    def __init__(self, model: nn.Module, absval):
        """
        Initialize a new ModelGraph object for converting a pytorch model to a graph we can perform graph algorithms on

        :param model: the model
        """
        super(ModelGraph, self).__init__()

        self.model = model
        self.absval = absval
        if model is not None:
            self._build_graph()

    def __str__(self):
        return f"ModelGraph()"

    def _get_weights(self):
        """
        Retrieve the weights from the model and pack them into a dictionary

        :return: a dictionary containing the weights
        """
        named_params = self.model.named_parameters()

        weights = {}

        for (key, value) in named_params:
            if 'weight' in key:
                weights[key] = value

        return weights

    def _build_graph(self):
        """
        Build the graph for the model
        """
        return NotImplementedError

    def _spectral_clustering(self, n_clusters=32, gamma=1.):
        """
        Performs network-wide spectral clustering.

        :return: the cluster labels
        """

        adj_matrix = nx.to_numpy_matrix(self)
        node_list = list(self.nodes())

        clusters = SpectralClustering(eigen_solver='arpack', n_init=100, affinity='precomputed', assign_labels="kmeans",
                                      n_clusters=n_clusters, gamma=gamma).fit_predict(adj_matrix)

        communities = [set() for _ in range(n_clusters)]

        for i, node in enumerate(node_list):
            label = clusters[i]
            communities[label].add(node)

        communities = [frozenset(i) for i in communities]

        return communities, clusters

    def get_model_modularity(self, n_clusters=8, resolution=1, method="louvain", partition_weights=True, q_weights="weight", second_graph=None):
        """
        Calculate the modularity of the model by first finding the best partitioning and then calculating the graph
        modularity.

        :param n_clusters The number of clusters used for cluster-based partitioning methods
        :param resolution The resolution for calculating the graph modularity (see the Resolution Limit Problem)
        :param method The partitioning method (spectral, greedy, louvain[preferred])

        :return: the model modularity
        """
        self.__class__ = nx.Graph
        if second_graph is not None:
            second_graph.__class__ = nx.Graph

        if method == "spectral":
            """
            Spectral clustering for graph partitioning. This needs a fix number of clusters.
            """
            communities, clusters = ModelGraph._spectral_clustering(self, n_clusters, gamma=resolution)

        elif method == "greedy":
            """
            Partitioning by greedily maximizing the graph modularity
            """
            communities = nx.community.greedy_modularity_communities(self, resolution=resolution)
            clusters = [0 for _ in range(len(self.nodes()))]

            for i, community in enumerate(communities):
                for j, node in enumerate(community):
                    clusters[node] = i

        elif method == "louvain":
            """
            Partitioning using the louvain algorithm
            """

            communities = nx.community.louvain_communities(self, resolution=resolution)
            clusters = [0 for _ in range(len(self.nodes()))]

            for i, community in enumerate(communities):
                for j, node in enumerate(community):
                    clusters[list(self.nodes()).index(node)] = i

        elif method == "girvan_newman":
            """
            Partitioning using the girvan-newman algorithm
            """
            def most_central_edge(G):
                centrality = betweenness(G, weight="weight")
                return max(centrality, key=centrality.get)

            if partition_weights:
                communities_generator = nx.community.girvan_newman(second_graph if second_graph is not None else self, most_valuable_edge=most_central_edge) # TOdo weights being used
            else:
                communities_generator = nx.community.girvan_newman(self)

            communities = []
            for com in next(communities_generator):
                communities.append(list(com))

            clusters = [0 for _ in range(len(self.nodes()))]

            for i, community in enumerate(communities):
                for j, node in enumerate(community):
                    clusters[list(self.nodes()).index(node)] = i

        else:
            raise Exception("Not a valid partitioning method (spectral, greedy, louvain[preferred])")

        return nx.algorithms.community.modularity(self, communities=communities, resolution=resolution, weight=q_weights), clusters  # Todo weight is now not none


class MLPGraph(ModelGraph):
    """
    Build a graph from an MLP. This enables us to perform graph algorithms on it.

    One important usage is the calculation of the network modularity using the model weights as edge weights.
    """

    def __str__(self):
        return f"MLPGraph()"

    def _build_graph(self):
        """
        Iteratively build the graph from the MLP using the weight matrices

        This uses the absolute values of the weight between two connected neurons as seen in
        https://arxiv.org/pdf/2110.08058.pdf
        """
        # get the named parameter weights
        weights = self._get_weights()

        # an iterator for iterating over the named parameters
        iterator = iter(weights.items())

        current_layer = 0

        while True:
            try:
                # get the current named parameter weight
                key, value = next(iterator)
                current_layer += 1
                for current_neuron in range(value.shape[1]):
                    current_node = f"layer{current_layer}:{current_neuron}"
                    # iterate over every sub node
                    for next_neuron in range(value.shape[0]):
                        next_node = f"layer{current_layer + 1}:{next_neuron}"

                        # add an edge between the two nodes using the absolute value of the parameter weight as the
                        # edge weight
                        self.add_node(current_node, layer=current_layer)
                        self.add_node(next_node, layer=current_layer + 1)

                        if self.absval:
                            edge = value[next_neuron, current_neuron].detach().abs().item()  # todo I removed abs()
                        else:
                            edge = value[next_neuron, current_neuron].detach().item()  # todo I removed abs()

                        if edge != 0:
                            self.add_edge(current_node, next_node, weight=edge)
            except StopIteration:
                break
