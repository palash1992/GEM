import os
import sys
import numpy as np
from subprocess import call
import matplotlib.pyplot as plt

from gem.embedding.static_graph_embedding import StaticGraphEmbedding
from gem.utils import graph_util
from gem.evaluation import visualize_embedding as viz


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

    def __init__(self, *hyper_dict, **kwargs):
        ''' Initialize the GraphFactorization class
        Args:
            d: dimension of the embedding
            eta: learning rate of sgd
            regu: regularization coefficient of magnitude of weights
            max_iter: max iterations in sgd
            print_step: #iterations to log the prgoress (step%print_step)
        '''
        hyper_params = {
            'print_step': 10000,
            'method_name': 'graph_factor_sgd'
        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])
        for dictionary in hyper_dict:
            for key in dictionary:
                self.__setattr__('_%s' % key, dictionary[key])

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self._d)

    def _get_f_value(self, graph):
        f1 = 0
        for i, j, w in graph.edges(data='weight', default=1):
            f1 += (w - np.dot(self._X[i, :], self._X[j, :]))**2
        f2 = self._regu * (np.linalg.norm(self._X)**2)
        return [f1, f2, f1 + f2]

    def learn_embedding(self, graph=None, edge_f=None,
                        is_weighted=False, no_python=True):
        c_flag = True
        if not graph and not edge_f:
            raise Exception('graph/edge_f needed')
        if no_python:
            if sys.platform[0] == "w":
                args = ["gem/c_exe/gf.exe"]
            else:
                args = ["gem/c_exe/gf"]
            if not graph and not edge_f:
                raise Exception('graph/edge_f needed')
            if edge_f:
                graph = graph_util.loadGraphFromEdgeListTxt(edge_f)
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
            try:
                call(args)
            except Exception as e:
                print(str(e))
                c_flag = False
                print('./gf not found. Reverting to Python implementation. Please compile gf, place node2vec in '
                      'the path and grant executable permission')
            if c_flag:
                try:
                    self._X = graph_util.loadEmbedding(emb_filename)
                except FileNotFoundError:
                    self._X = np.random.randn(len(graph.nodes), self._d)
                try:
                    call(["rm", emb_filename])
                except:
                    pass
                return self._X
        if not graph:
            graph = graph_util.loadGraphFromEdgeListTxt(edge_f)
        self._node_num = len(graph.nodes)
        self._X = 0.01 * np.random.randn(self._node_num, self._d)
        for iter_id in range(self._max_iter):
            if not iter_id % self._print_step:
                [f1, f2, f] = self._get_f_value(graph)
                print('\t\tIter id: %d, Objective: %g, f1: %g, f2: %g' % (
                    iter_id,
                    f,
                    f1,
                    f2
                ))
            for i, j, w in graph.edges(data='weight', default=1):
                if j <= i:
                    continue
                term1 = -(w - np.dot(self._X[i, :], self._X[j, :])) * self._X[j, :]
                term2 = self._regu * self._X[i, :]
                delPhi = term1 + term2
                self._X[i, :] -= self._eta * delPhi
        return self._X

    def get_embedding(self):
        return self._X

    def get_edge_weight(self, i, j):
        return np.dot(self._X[i, :], self._X[j, :])
