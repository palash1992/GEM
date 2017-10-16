disp_avlbl = True
import os
if 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
import scipy.io as sio

import sys
sys.path.append('./')
sys.path.append(os.path.realpath(__file__))

from .static_graph_embedding import StaticGraphEmbedding
from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from time import time
		
class GraphFactorization(StaticGraphEmbedding):

	def __init__(self, d, max_iter, eta, regu):
		''' Initialize the GraphFactorization class

		Args:
			d: dimension of the embedding
			eta: learning rate of sgd
			regu: regularization coefficient of magnitude of weights
			max_iter: max iterations in sgd
		'''
		self._d = d
		self._eta = eta
		self._regu = regu
		self._max_iter = max_iter
		self._method_name = 'graph_factor_sgd'

	def get_method_name(self):
		return self._method_name

	def get_method_summary(self):
		return '%s_%d' % (self._method_name, self._d)

	def _get_f_value(self, graph):
		f1 = 0
		for i, j, w in graph.edges(data='weight', default=1):
			f1 += (w - np.dot(self._X[i, :], self._X[j, :]))**2
		f2 = self._regu*(np.linalg.norm(self._X)**2)
		return [f1, f2, f1+f2]

	def learn_embedding(self, graph=None, edge_f=None, is_weighted=False, no_python=False):
		c_flag = True
		if not graph and not edge_f:
			raise Exception('graph/edge_f needed')
		if no_python:
			try:
				from c_ext import graphFac_ext
			except:
				print('Could not import C++ module for Graph Factorization. Reverting to python implementation. Please recompile graphFac_ext from graphFac.cpp using bjam')
				c_flag = False
			if c_flag:
				if edge_f:
					graph = graph_util.loadGraphFromEdgeListTxt(edge_f)
				graph_util.saveGraphToEdgeListTxt(graph, 'tempGraph.graph')
				is_weighted = True
				edge_f = 'tempGraph.graph'
				t1 = time()
				graphFac_ext.learn_embedding(edge_f, "tempGraphGF.emb", True, is_weighted, self._d, self._eta, self._regu, self._max_iter)	
				self._X = graph_util.loadEmbedding('tempGraphGF.emb')
				t2 = time()
				return self._X, (t2-t1)
				
		if not graph:
			graph = graph_util.loadGraphFromEdgeListTxt(edge_f)
		t1 = time()
		self._node_num = graph.number_of_nodes()
		self._X = 0.01*np.random.randn(self._node_num, self._d)
		for iter_id in range(self._max_iter):
			if not iter_id%100:
				[f1, f2, f] = self._get_f_value(graph)
				print('\t\tIter id: %d, Objective value: %g, f1: %g, f2: %g' % (iter_id, f, f1, f2))
			tempFlag = False
			for i, j, w in graph.edges(data='weight', default=1):
				if j <= i:
					continue
				delPhi = -(w - np.dot(self._X[i, :], self._X[j, :]))*self._X[j, :] + self._regu*self._X[i, :]
				self._X[i, :] -= self._eta*delPhi
		t2 = time()
		return self._X, (t2 - t1)

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
	print ('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
	t1 = time()
	embedding = GraphFactorization(2, 100000, 1*10**-4, 1.0)
	embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
	print ('Graph Factorization:\n\tTraining time: %f' % (time() - t1))

	viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=None)
	plt.show()

