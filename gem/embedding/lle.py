import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from sklearn.preprocessing import normalize

from gem.embedding.static_graph_embedding import StaticGraphEmbedding


class LocallyLinearEmbedding(StaticGraphEmbedding):
    hyper_params = {
        'method_name': 'lle_svd'
    }

    def __init__(self, *args, **kwargs):
        """ Initialize the LocallyLinearEmbedding class

        Args:
            d: dimension of the embedding
        """
        super(LocallyLinearEmbedding, self).__init__(*args, **kwargs)

    def learn_embedding(self, graph=None,
                        is_weighted=False, no_python=False):
        if not graph:
            raise Exception('graph needed')
        graph = graph.to_undirected()
        A = nx.to_scipy_sparse_matrix(graph)
        normalize(A, norm='l1', axis=1, copy=False)
        I_n = sp.eye(len(graph.nodes))
        I_min_A = I_n - A
        u, s, vt = lg.svds(I_min_A, k=self._d + 1, which='SM')
        self._X = vt.T
        self._X = self._X[:, 1:]
        return self._X.real

    def get_edge_weight(self, i, j):
        return np.exp(
            -np.power(np.linalg.norm(self._X[i, :] - self._X[j, :]), 2)
        )
