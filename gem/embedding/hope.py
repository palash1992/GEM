import networkx as nx
import numpy as np
import scipy.sparse.linalg as lg
from gem.embedding.static_graph_embedding import StaticGraphEmbedding
from gem.utils import graph_util


class HOPE(StaticGraphEmbedding):

    hyper_params = {
        'method_name': 'hope_gsvd'
    }

    def __init__(self, *args, **kwargs):
        """ Initialize the HOPE class

        Args:
            d: dimension of the embedding
            beta: higher order coefficient
        """
        super(HOPE, self).__init__(*args, **kwargs)

    def learn_embedding(self, graph=None,
                        is_weighted=False, no_python=False):
        if not graph:
            raise ValueError('graph needed')

        A = nx.to_numpy_array(graph)
        m_g = np.eye(len(graph.nodes)) - self._beta * A
        m_l = self._beta * A
        S = np.dot(np.linalg.inv(m_g), m_l)

        u, s, vt = lg.svds(S, k=self._d // 2)
        X1 = np.dot(u, np.diag(np.sqrt(s)))
        X2 = np.dot(vt.T, np.diag(np.sqrt(s)))
        self._X = np.concatenate((X1, X2), axis=1)

        p_d_p_t = np.dot(u, np.dot(np.diag(s), vt))
        eig_err = np.linalg.norm(p_d_p_t - S)
        print('SVD error (low rank): %f' % eig_err)
        return self._X

    def get_edge_weight(self, i, j):
        return np.dot(self._X[i, :self._d // 2], self._X[j, self._d // 2:])
