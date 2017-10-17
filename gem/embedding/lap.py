disp_avlbl = True
import os
if 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from time import time

import sys
sys.path.append('./')
sys.path.append(os.path.realpath(__file__))

from .static_graph_embedding import StaticGraphEmbedding
from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz

class LaplacianEigenmaps(StaticGraphEmbedding):

	def __init__(self, d):
		self._d = d
		self._method_name = 'lap_eigmap_svd'
		self._X = None

	def get_method_name(self):
		return self._method_name

	def get_method_summary(self):
		return '%s_%d' % (self._method_name, self._d)

	def learn_embedding(self, graph=None, edge_f=None, is_weighted=False, no_python=False):
		# A = nx.to_scipy_sparse_matrix(G)
		# if not np.allclose(A.T, A):
		# 	print "laplace eigmap approach only works for symmetric graphs!"
		# 	return

		# self._node_num = A.shape[0]

		# D = np.diag(np.sum(A, 1))
		# L_G = D - A
		# zeroRows = np.where(D.sum(1)==0)
		# D[zeroRows, zeroRows] = np.inf
		# d_min_half = np.linalg.inv(np.sqrt(D))
		# L_sym = np.dot(d_min_half, np.dot(L_G, d_min_half))
		if not graph and not edge_f:
			raise Exception('graph/edge_f needed')
		if not graph:
			graph = graph_util.loadGraphFromEdgeListTxt(edge_f)
		graph = graph.to_undirected()
		t1 = time()
		L_sym = nx.normalized_laplacian_matrix(graph)

		w, v = lg.eigs(L_sym, k=self._d+1, which='SM')
		t2 = time()
		self._X = v[:, 1:]

		# p_d_p_t = np.dot(v, np.dot(np.diag(w), v.T))
		# eig_err = np.linalg.norm(p_d_p_t - L_sym)
		# print 'Laplacian matrix reconstruction error (low rank): %f' % eig_err

		# p_d_p_t = np.dot(self._X, np.dot(w[1:self._d+1, 1:self._d+1], self._X.T))
		# eig_err = np.linalg.norm(p_d_p_t - L_sym)
		# print 'Laplacian reconstruction error (low rank approx): %f' % eig_err
		return self._X, (t2-t1)

	def get_embedding(self):
		return self._X

	def get_edge_weight(self, i, j):
		return np.exp(-np.power(np.linalg.norm(self._X[i, :] - self._X[j, :]), 2))

	def get_reconstructed_adj(self, X=None, node_l=None):
		if X is not None:
			node_num = X.shape[0]
			self._X = X
		else:
			node_num = self._node_num
		adj_mtx_r = np.zeros((node_num, node_num)) # G_r is the reconstructed graph
		for v_i in range(node_num):
			for v_j in range(node_num):
				if v_i == v_j:
					continue
				adj_mtx_r[v_i, v_j] = self.get_edge_weight(v_i, v_j)
		return adj_mtx_r

if __name__ == '__main__':
	# load Zachary's Karate graph
	edge_f = 'data/karate.edgelist'
	G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=False)
	G = G.to_directed()
	res_pre = 'results/testKarate'
	print('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
	t1 = time()
	embedding = LaplacianEigenmaps(2)
	embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
	print('Laplacian Eigenmaps:\n\tTraining time: %f' % (time() - t1))1

	viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=None)
	plt.show()

