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

from subprocess import call

from .static_graph_embedding import StaticGraphEmbedding
from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz

class node2vec(StaticGraphEmbedding):

	def __init__(self, d, max_iter, wLen, nWalks, cSize, ret_p, inout_p):
		self._d = d
		self._max_iter = max_iter
		self._walkLength = wLen
		self._numWalks = nWalks
		self._contextSize = cSize
		self._return_p = ret_p
		self._inout_p = inout_p

		self._method_name = 'node2vec_rw'
		self._X = None

	def get_method_name(self):
		return self._method_name

	def get_method_summary(self):
		return '%s_%d' % (self._method_name, self._d)

	def learn_embedding(self, graph=None, edge_f=None, is_weighted=False, no_python=False):
		args = ["./c_exe/node2vec"]
		if not graph and not edge_f:
			raise Exception('graph/edge_f needed')
		if edge_f:
			graph = graph_util.loadGraphFromEdgeListTxt(edge_f)
		graph_util.saveGraphToEdgeListTxtn2v(graph, 'tempGraph.graph')
		args.append("-i:tempGraph.graph")
		args.append("-o:tempGraph.emb")
		args.append("-d:%d" % self._d)
		args.append("-l:%d" % self._walkLength)
		args.append("-r:%d" % self._numWalks)
		args.append("-k:%d" % self._contextSize)
		args.append("-e:%d" % self._max_iter)
		args.append("-p:%f" % self._return_p)
		args.append("-q:%f" % self._inout_p)
		args.append("-v")
		args.append("-dr")
		args.append("-w")
		t1 = time()
		try:
			call(args)
		except:
			raise Exception('./node2vec not found. Please compile snap, place node2vec in the path and grant executable permission')
		self._X = graph_util.loadEmbedding('tempGraph.emb')
		t2 = time()
		return self._X, (t2-t1)

	def get_embedding(self):
		return self._X

	def get_edge_weight(self, i, j):
		return np.dot(self._X[i, :], self._X[j, :])
		
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
	embedding = node2vec(2, 1, 80, 10, 10, 1, 1)
	embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
	print('node2vec:\n\tTraining time: %f' % (time() - t1))

	viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=None)
	plt.show()

