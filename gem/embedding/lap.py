import networkx as nx
import numpy as np
import scipy.sparse.linalg as lg

from gem.embedding.static_graph_embedding import StaticGraphEmbedding
from gem.utils import graph_util


class LaplacianEigenmaps(StaticGraphEmbedding):

    def __init__(self, *hyper_dict, **kwargs):
        ''' Initialize the LaplacianEigenmaps class

        Args:
            d: dimension of the embedding
        '''
        hyper_params = {
            'method_name': 'lap_eigmap_svd'
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
        L_sym = nx.normalized_laplacian_matrix(graph)

        w, v = lg.eigs(L_sym, k=self._d + 1, which='SM')
        idx = np.argsort(w)  # sort eigenvalues
        w = w[idx]
        v = v[:, idx]
        self._X = v[:, 1:]

        p_d_p_t = np.dot(v, np.dot(np.diag(w), v.T))
        eig_err = np.linalg.norm(p_d_p_t - L_sym)
        print('Laplacian matrix recon. error (low rank): %f' % eig_err)
        return self._X.real

    def get_embedding(self):
        return self._X

    def get_edge_weight(self, i, j):
        return np.exp(
            -np.power(np.linalg.norm(self._X[i, :] - self._X[j, :]), 2)
        )
