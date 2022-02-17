import networkx as nx
import numpy as np
import scipy.sparse.linalg as lg
from gem.embedding.static_graph_embedding import StaticGraphEmbedding
from gem.utils import graph_util


class HOPE(StaticGraphEmbedding):

    def __init__(self, *hyper_dict, **kwargs):
        """ Initialize the HOPE class

        Args:
            d: dimension of the embedding
            beta: higher order coefficient
        """
        hyper_params = {
            'method_name': 'hope_gsvd'
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

        A = nx.to_numpy_matrix(graph)
        M_g = np.eye(len(graph.nodes)) - self._beta * A
        M_l = self._beta * A
        S = np.dot(np.linalg.inv(M_g), M_l)

        u, s, vt = lg.svds(S, k=self._d // 2)
        X1 = np.dot(u, np.diag(np.sqrt(s)))
        X2 = np.dot(vt.T, np.diag(np.sqrt(s)))
        self._X = np.concatenate((X1, X2), axis=1)

        p_d_p_t = np.dot(u, np.dot(np.diag(s), vt))
        eig_err = np.linalg.norm(p_d_p_t - S)
        print('SVD error (low rank): %f' % eig_err)
        return self._X

    def get_embedding(self):
        return self._X

    def get_edge_weight(self, i, j):
        return np.dot(self._X[i, :self._d // 2], self._X[j, self._d // 2:])
