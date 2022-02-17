import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from sklearn.preprocessing import normalize

from gem.embedding.static_graph_embedding import StaticGraphEmbedding
from gem.utils import graph_util


class LocallyLinearEmbedding(StaticGraphEmbedding):

    def __init__(self, *hyper_dict, **kwargs):
        ''' Initialize the LocallyLinearEmbedding class

        Args:
            d: dimension of the embedding
        '''
        hyper_params = {
            'method_name': 'lle_svd'
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

    def learn_embedding(self, graph=None, edge_f=None,
                        is_weighted=False, no_python=False):
        if not graph and not edge_f:
            raise Exception('graph/edge_f needed')
        if not graph:
            graph = graph_util.loadGraphFromEdgeListTxt(edge_f)
        graph = graph.to_undirected()
        A = nx.to_scipy_sparse_matrix(graph)
        normalize(A, norm='l1', axis=1, copy=False)
        I_n = sp.eye(len(graph.nodes))
        I_min_A = I_n - A
        u, s, vt = lg.svds(I_min_A, k=self._d + 1, which='SM')
        self._X = vt.T
        self._X = self._X[:, 1:]
        return self._X.real

    def get_embedding(self):
        return self._X

    def get_edge_weight(self, i, j):
        return np.exp(
            -np.power(np.linalg.norm(self._X[i, :] - self._X[j, :]), 2)
        )
