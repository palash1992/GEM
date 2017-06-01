disp_avlbl = True
from os import environ
if 'DISPLAY' not in environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.io as sio

import sys
sys.path.append('./')

from static_graph_embedding import StaticGraphEmbedding
from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from sdne_utils import *

from keras.layers import Input, Dense, Lambda, merge
from keras.models import Model, model_from_json
import keras.regularizers as Reg
from keras.optimizers import SGD, Adam
from keras import backend as KBack

from theano.printing import debugprint as dbprint, pprint
from time import time

class SDNE(StaticGraphEmbedding):

	def __init__(self, d, beta, alpha, nu1, nu2, K, n_units, rho, n_iter, xeta, n_batch, modelfile=None, weightfile=None, node_frac=1, n_walks_per_node=5, len_rw=2, savefilesuffix=None, subsample=False):
		''' Initialize the SDNE class

		Args:
			d: dimension of the embedding
			beta: penalty parameter in matrix B of 2nd order objective
			alpha: weighing hyperparameter for 1st order objective
			nu1: L1-reg hyperparameter
			nu2: L2-reg hyperparameter
			K: number of hidden layers in encoder/decoder
			n_units: vector of length K-1 containing #units in hidden layers of encoder/decoder, not including the units in the embedding layer
			rho: bounding ratio for number of units in consecutive layers (< 1)
			n_iter: number of sgd iterations for first embedding (const)
			n_iter_subs: number of sgd iterations for subsequent embeddings (const)
			xeta: sgd step size parameter
			n_batch: minibatch size for SGD
			modelfile: Files containing previous encoder and decoder models
			weightfile: Files containing previous encoder and decoder weights
			node_frac: Fraction of nodes to use for random walk
			n_walks_per_node: Number of random walks to do for each selected nodes
			len_rw: Length of every random walk
		'''
		self._method_name = 'sdne' # embedding method name
		self._d = d
		self._Y = None # embedding
		self._beta = beta
		self._alpha = alpha
		self._nu1 = nu1
		self._nu2 = nu2
		self._K = K
		self._n_units = n_units
		self._actfn = 'relu' # We use relu instead of sigmoid from the paper, to avoid vanishing gradients and allow correct layer deepening
		self._rho = rho
		self._n_iter = n_iter
		self._xeta = xeta
		self._n_batch = n_batch
		self._modelfile = modelfile
		self._weightfile = weightfile
		self._node_frac = node_frac
		self._n_walks_per_node = n_walks_per_node
		self._len_rw = len_rw
		self._savefilesuffix = savefilesuffix
		self._subsample = subsample
		self._num_iter = n_iter # max number of iterations during sgd (variable)		
		# self._node_num is number of nodes: initialized later in learn_embedding()
		# self._encoder is the vertex->embedding model
		# self._decoder is the embedding->vertex model
		# self._autocoder is the vertex->(vertex,embedding) model
		# self._model is the SDNE model to be trained (uses self._autoencoder internally)
		

	def get_method_name(self):
		return self._method_name

	def get_method_summary(self):
		return '%s_%d' % (self._method_name, self._d)

	def learn_embedding(self, graph=None, edge_f=None, is_weighted=False, no_python=False):
		if not graph and not edge_f:
			raise Exception('graph/edge_f needed')
		if not graph:
			graph = graph_util.loadGraphFromEdgeListTxt(edge_f)
		if self._subsample:
			S = graph_util.randwalk_DiGraph_to_adj(graph, node_frac=self._node_frac, n_walks_per_node=self._n_walks_per_node, len_rw=self._len_rw)
		else:
			S = graph_util.transform_DiGraph_to_adj(graph)
		if not np.allclose(S.T, S):
			print "SDNE only works for symmetric graphs! Making the graph symmetric"
		t1 = time()
		S = (S + S.T)/2					# enforce S is symmetric
		S -= np.diag(np.diag(S))		# enforce diagonal = 0
		self._node_num = S.shape[0]		
		n_edges = np.count_nonzero(S)	# Double counting symmetric edges deliberately to maintain autoencoder symmetry
		# Create matrix B
		B = np.ones(S.shape)
		B[S != 0] = self._beta

		# compute degree of each node
		deg = np.sum(S!=0, 1)

		
		# Generate encoder, decoder and autoencoder
		self._num_iter = self._n_iter
		# If cannot use previous step information, initialize new models
		self._encoder = get_encoder(self._node_num, self._d, self._K, self._n_units, self._nu1, self._nu2, self._actfn)
		self._decoder = get_decoder(self._node_num, self._d, self._K, self._n_units, self._nu1, self._nu2, self._actfn)
		self._autoencoder = get_autoencoder(self._encoder, self._decoder)

		# Initialize self._model
		# Input	
		x_in = Input(shape=(2*self._node_num,), name='x_in')
		x1 = Lambda(lambda x: x[:,0:self._node_num], output_shape=(self._node_num,))(x_in)
		x2 = Lambda(lambda x: x[:,self._node_num:2*self._node_num], output_shape=(self._node_num,))(x_in)
		# Process inputs
		[x_hat1, y1] = self._autoencoder(x1)
		[x_hat2, y2] = self._autoencoder(x2)
		# Outputs
		x_diff1 = merge([x_hat1, x1], mode=lambda (a,b): a - b, output_shape=lambda L: L[1])
		x_diff2 = merge([x_hat2, x2], mode=lambda (a,b): a - b, output_shape=lambda L: L[1])
		y_diff = merge([y2, y1], mode=lambda (a,b): a - b, output_shape=lambda L: L[1])

		# Objectives
		def weighted_mse_x(y_true, y_pred):
			''' Hack: This fn doesn't accept additional arguments. We use y_true to pass them.
				y_pred: Contains x_hat - x
				y_true: Contains [b, deg]
			'''			
			return KBack.sum(KBack.square(y_pred * y_true[:,0:self._node_num]), axis=-1)/y_true[:,self._node_num]
		def weighted_mse_y(y_true, y_pred):
			''' Hack: This fn doesn't accept additional arguments. We use y_true to pass them.
			y_pred: Contains y2 - y1
			y_true: Contains s12
			'''
			min_batch_size = KBack.shape(y_true)[0]        
			return KBack.reshape(KBack.sum(KBack.square(y_pred), axis=-1),[min_batch_size, 1]) * y_true
		
		# Model
		self._model = Model(input=x_in, output=[x_diff1, x_diff2, y_diff])
		sgd = SGD(lr=self._xeta, decay=1e-5, momentum=0.99, nesterov=True)
		# adam = Adam(lr=self._xeta, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
		self._model.compile(optimizer=sgd, loss=[weighted_mse_x, weighted_mse_x, weighted_mse_y], loss_weights=[1, 1, self._alpha])
		# self._model.compile(optimizer=adam, loss=[weighted_mse_x, weighted_mse_x, weighted_mse_y], loss_weights=[1, 1, self._alpha])

		# Structure data in the correct format for the SDNE model
		# InData format: [x1, x2]
		# OutData format: [b1, b2, s12, deg1, deg2]
		data_chunk_size = 100000
		InData = np.zeros((data_chunk_size, 2*self._node_num))
		OutData = np.zeros((data_chunk_size, 2*self._node_num + 3))
		# Train the model
		for epoch_num in range(self._num_iter):			
			print 'EPOCH %d/%d' % (epoch_num, self._num_iter)
			e = 0
			k = 0
			for i in range(self._node_num):
				for j in range(self._node_num):
					if(S[i,j] != 0):
						temp = np.append(S[i,:], S[j,:])
						InData[k,:] = temp
						temp = np.append(np.append(np.append(np.append(B[i,:], B[j,:]), S[i,j]), deg[i]), deg[j])
						OutData[k,:] = temp
						e += 1		
						k += 1
						if k == data_chunk_size:
							self._model.fit(InData, [ np.append(OutData[:,0:self._node_num], np.reshape(OutData[:,2*self._node_num+1], [data_chunk_size, 1]), axis=1), 
								np.append(OutData[:,self._node_num:2*self._node_num], np.reshape(OutData[:,2*self._node_num+2], [data_chunk_size, 1]), axis=1), 
								OutData[:,2*self._node_num] ], nb_epoch=1, batch_size=self._n_batch, shuffle=True, verbose=1)
							k = 0
			if k > 0:
				self._model.fit(InData[:k, :], [ np.append(OutData[:k,0:self._node_num], np.reshape(OutData[:k,2*self._node_num+1], [k, 1]), axis=1), 
					np.append(OutData[:k,self._node_num:2*self._node_num], np.reshape(OutData[:k,2*self._node_num+2], [k, 1]), axis=1), 
					OutData[:k,2*self._node_num] ], nb_epoch=1, batch_size=self._n_batch, shuffle=True, verbose=1) 

		# Get embedding for all points
		_, self._Y = self._autoencoder.predict(S, batch_size=self._n_batch)
		t2 = time()
		# Save the autoencoder and its weights
		if(self._weightfile is not None):
			saveweights(self._encoder, self._weightfile[0])
			saveweights(self._decoder, self._weightfile[1])
		if(self._modelfile is not None):
			savemodel(self._encoder, self._modelfile[0])
			savemodel(self._decoder, self._modelfile[1])		
		if(self._savefilesuffix is not None):
			saveweights(self._encoder, 'encoder_weights_'+self._savefilesuffix+'.hdf5')
			saveweights(self._decoder, 'decoder_weights_'+self._savefilesuffix+'.hdf5')
			savemodel(self._encoder, 'encoder_model_'+self._savefilesuffix+'.json')
			savemodel(self._decoder, 'decoder_model_'+self._savefilesuffix+'.json')
			# Save the embedding
			np.savetxt('embedding_'+self._savefilesuffix+'.txt', self._Y)		
		return self._Y, (t2-t1)


	def get_embedding(self, filesuffix):
		return self._Y if filesuffix is None else np.loadtxt('embedding_'+filesuffix+'.txt')


	def get_edge_weight(self, i, j, embed=None, filesuffix=None):
		if embed is None:
			if filesuffix is None:
				embed = self._Y
			else:
				embed = np.loadtxt('embedding_'+filesuffix+'.txt')
		if i == j:
			return 0
		else:
			S_hat = self.get_reconst_from_embed(embed[(i,j),:], filesuffix)
			return (S_hat[i,j] + S_hat[j,i])/2

	def get_reconstructed_adj(self, embed=None, node_l=None, filesuffix=None):
		if embed is None:
			if filesuffix is None:
				embed = self._Y
			else:
				embed = np.loadtxt('embedding_'+filesuffix+'.txt')
		S_hat = self.get_reconst_from_embed(embed, node_l, filesuffix)		
		return graphify(S_hat)

	def get_reconst_from_embed(self, embed, node_l=None, filesuffix=None):
		if filesuffix is None:
			if node_l is not None:
				return self._decoder.predict(embed, batch_size=self._n_batch)[:, node_l]
			else:
				return self._decoder.predict(embed, batch_size=self._n_batch)
		else:
			try:
				decoder = model_from_json(open('decoder_model_'+filesuffix+'.json').read())
			except:
				print('Error reading file: {0}. Cannot load previous model'.format('decoder_model_'+filesuffix+'.json'))
				exit()
			try:
				decoder.load_weights('decoder_weights_'+filesuffix+'.hdf5')
			except:
				print('Error reading file: {0}. Cannot load previous weights'.format('decoder_weights_'+filesuffix+'.hdf5'))
				exit()
			if node_l is not None:
				return decoder.predict(embed, batch_size=self._n_batch)[:, node_l]
			else:
				return decoder.predict(embed, batch_size=self._n_batch)

if __name__ == '__main__':
	# load Zachary's Karate graph
	edge_f = 'data/karate.edgelist'
	G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=False)
	G = G.to_directed()
	res_pre = 'results/testKarate'
	print 'Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges())
	t1 = time()
	embedding = SDNE(d=2, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3, n_units=[50, 15,], rho=0.3, n_iter=50, xeta=0.01, n_batch=500, modelfile=['./intermediate/enc_model.json', './intermediate/dec_model.json'], weightfile=['./intermediate/enc_weights.hdf5', './intermediate/dec_weights.hdf5'])
	embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
	print 'SDNE:\n\tTraining time: %f' % (time() - t1)

	viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=None)
	plt.show()