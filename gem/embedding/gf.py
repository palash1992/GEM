import os
import sys
import numpy as np
from subprocess import call
import matplotlib.pyplot as plt

from gem.embedding.static_graph_embedding import StaticGraphEmbedding
from gem.utils import graph_util


class GraphFactorization(StaticGraphEmbedding):

    """`Graph Factorization`_.
    Graph Factorization factorizes the adjacency matrix with regularization.

    Args:
        hyper_dict (object): Hyper parameters.
        kwargs (dict): keyword arguments, form updating the parameters

    Examples:
        >>> from gem.embedding.gf import GraphFactorization
        >>> edge_f = 'data/karate.edgelist'
        >>> G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=False)
        >>> G = G.to_directed()
        >>> res_pre = 'results/testKarate'
        >>> graph_util.print_graph_stats(G)
        >>> t1 = time()
        >>> embedding = GraphFactorization(2, 100000, 1 * 10**-4, 1.0)
        >>> embedding.learn_embedding(graph=G, edge_f=None,
                                  is_weighted=True, no_python=True)
        >>> print ('Graph Factorization:Training time: %f' % (time() - t1))
        >>> viz.plot_embedding2D(embedding.get_embedding(),
                             di_graph=G, node_colors=None)
        >>> plt.show()
    .. _Graph Factorization:
        https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40839.pdf
    """
    hyper_params = {
        'print_step': 10000,
        'method_name': 'graph_factor_sgd'
    }

    def __init__(self, *args, **kwargs):
        """ Initialize the GraphFactorization class
        Args:
            d: dimension of the embedding
            eta: learning rate of sgd
            regu: regularization coefficient of magnitude of weights
            max_iter: max iterations in sgd
            print_step: #iterations to log the prgoress (step%print_step)
        """
        super(GraphFactorization, self).__init__(*args, **kwargs)

    def _use_c_implementation(self, graph=None):
        if sys.platform[0] == "w":
            args = ["gem/c_exe/gf.exe"]
        else:
            args = ["gem/c_exe/gf"]
        os.makedirs('gem/intermediate', exist_ok=True)
        graph_filename = 'gem/intermediate/%s_gf.graph' % self._data_set
        emb_filename = 'gem/intermediate/%s_%d_gf.emb' % (self._data_set, self._d)
        graph_util.saveGraphToEdgeListTxt(graph, graph_filename)
        args.append(graph_filename)
        args.append(emb_filename)
        args.append("1")  # Verbose
        args.append("1")  # Weighted
        args.append("%d" % self._d)
        args.append("%f" % self._eta)
        args.append("%f" % self._regu)
        args.append("%d" % self._max_iter)
        args.append("%d" % self._print_step)
        call(args)
        try:
            self._X = graph_util.loadEmbedding(emb_filename)
        except FileNotFoundError:
            self._X = np.random.randn(len(graph.nodes), self._d)
        os.remove(emb_filename)
        return self._X

    def learn_embedding(self, graph=None,
                        is_weighted=False, no_python=True):
        if not graph:
            raise Exception('graph/edge_f needed')
        if no_python:
            try:
                self._use_c_implementation(graph)
            except FileNotFoundError:
                print('./gf not found. Reverting to Python implementation. Please compile gf, place node2vec in '
                      'the path and grant executable permission')
        self._node_num = len(graph.nodes)
        self._X = 0.01 * np.random.randn(self._node_num, self._d)
        for iter_id in range(self._max_iter):
            for i, j, w in graph.edges(data='weight', default=1):
                if j <= i:
                    continue
                term1 = -(w - np.dot(self._X[i, :], self._X[j, :])) * self._X[j, :]
                term2 = self._regu * self._X[i, :]
                delPhi = term1 + term2
                self._X[i, :] -= self._eta * delPhi
        return self._X

    def get_edge_weight(self, i, j):
        return np.dot(self._X[i, :], self._X[j, :])
