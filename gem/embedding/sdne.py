disp_avlbl = True
import os
if 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.io as sio
import networkx as nx

import sys
sys.path.append('./')
sys.path.append(os.path.realpath(__file__))

from .static_graph_embedding import StaticGraphEmbedding
from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from .sdne_utils import *

from keras.layers import Input, Dense, Lambda, merge
from keras.models import Model, model_from_json
import keras.regularizers as Reg
from keras.optimizers import SGD, Adam
from keras import backend as KBack

from time import time


class SDNE(StaticGraphEmbedding):

    def __init__(self, *hyper_dict, **kwargs):
        ''' Initialize the SDNE class

        Args:
            d: dimension of the embedding
            beta: penalty parameter in matrix B of 2nd order objective
            alpha: weighing hyperparameter for 1st order objective
            nu1: L1-reg hyperparameter
            nu2: L2-reg hyperparameter
            K: number of hidden layers in encoder/decoder
            n_units: vector of length K-1 containing #units in hidden layers
                     of encoder/decoder, not including the units in the
                     embedding layer
            rho: bounding ratio for number of units in consecutive layers (< 1)
            n_iter: number of sgd iterations for first embedding (const)
            xeta: sgd step size parameter
            n_batch: minibatch size for SGD
            modelfile: Files containing previous encoder and decoder models
            weightfile: Files containing previous encoder and decoder weights
        '''
        hyper_params = {
            'method_name': 'sdne',
            'actfn': 'relu',
            'modelfile': None,
            'weightfile': None,
            'savefilesuffix': None

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
        S = nx.to_scipy_sparse_matrix(graph)
        t1 = time()
        S = (S + S.T) / 2
        self._node_num = graph.number_of_nodes()

        # Generate encoder, decoder and autoencoder
        self._num_iter = self._n_iter
        # If cannot use previous step information, initialize new models
        self._encoder = get_encoder(self._node_num, self._d,
                                    self._K, self._n_units,
                                    self._nu1, self._nu2,
                                    self._actfn)
        self._decoder = get_decoder(self._node_num, self._d,
                                    self._K, self._n_units,
                                    self._nu1, self._nu2,
                                    self._actfn)
        self._autoencoder = get_autoencoder(self._encoder, self._decoder)

        # Initialize self._model
        # Input
        x_in = Input(shape=(2 * self._node_num,), name='x_in')
        x1 = Lambda(
            lambda x: x[:, 0:self._node_num],
            output_shape=(self._node_num,)
        )(x_in)
        x2 = Lambda(
            lambda x: x[:, self._node_num:2 * self._node_num],
            output_shape=(self._node_num,)
        )(x_in)
        # Process inputs
        [x_hat1, y1] = self._autoencoder(x1)
        [x_hat2, y2] = self._autoencoder(x2)
        # Outputs
        x_diff1 = merge([x_hat1, x1],
                        mode=lambda ab: ab[0] - ab[1],
                        output_shape=lambda L: L[1])
        x_diff2 = merge([x_hat2, x2],
                        mode=lambda ab: ab[0] - ab[1],
                        output_shape=lambda L: L[1])
        y_diff = merge([y2, y1],
                       mode=lambda ab: ab[0] - ab[1],
                       output_shape=lambda L: L[1])

        # Objectives
        def weighted_mse_x(y_true, y_pred):
            ''' Hack: This fn doesn't accept additional arguments.
                      We use y_true to pass them.
                y_pred: Contains x_hat - x
                y_true: Contains [b, deg]
            '''
            return KBack.sum(
                KBack.square(y_pred * y_true[:, 0:self._node_num]),
                axis=-1) / y_true[:, self._node_num]

        def weighted_mse_y(y_true, y_pred):
            ''' Hack: This fn doesn't accept additional arguments.
                      We use y_true to pass them.
            y_pred: Contains y2 - y1
            y_true: Contains s12
            '''
            min_batch_size = KBack.shape(y_true)[0]
            return KBack.reshape(
                KBack.sum(KBack.square(y_pred), axis=-1),
                [min_batch_size, 1]
            ) * y_true

        # Model
        self._model = Model(input=x_in, output=[x_diff1, x_diff2, y_diff])
        sgd = SGD(lr=self._xeta, decay=1e-5, momentum=0.99, nesterov=True)
        # adam = Adam(lr=self._xeta, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self._model.compile(
            optimizer=sgd,
            loss=[weighted_mse_x, weighted_mse_x, weighted_mse_y],
            loss_weights=[1, 1, self._alpha]
        )

        self._model.fit_generator(
            generator=batch_generator_sdne(S, self._beta, self._n_batch, True),
            nb_epoch=self._num_iter,
            samples_per_epoch=S.nonzero()[0].shape[0] // self._n_batch,
            verbose=1
        )
        # Get embedding for all points
        self._Y = model_batch_predictor(self._autoencoder, S, self._n_batch)
        t2 = time()
        # Save the autoencoder and its weights
        if(self._weightfile is not None):
            saveweights(self._encoder, self._weightfile[0])
            saveweights(self._decoder, self._weightfile[1])
        if(self._modelfile is not None):
            savemodel(self._encoder, self._modelfile[0])
            savemodel(self._decoder, self._modelfile[1])
        if(self._savefilesuffix is not None):
            saveweights(
                self._encoder,
                'encoder_weights_' + self._savefilesuffix + '.hdf5'
            )
            saveweights(
                self._decoder,
                'decoder_weights_' + self._savefilesuffix + '.hdf5'
            )
            savemodel(
                self._encoder,
                'encoder_model_' + self._savefilesuffix + '.json'
            )
            savemodel(
                self._decoder,
                'decoder_model_' + self._savefilesuffix + '.json'
            )
            # Save the embedding
            np.savetxt('embedding_' + self._savefilesuffix + '.txt', self._Y)
        return self._Y, (t2 - t1)

    def get_embedding(self, filesuffix=None):
        return self._Y if filesuffix is None else np.loadtxt(
            'embedding_' + filesuffix + '.txt'
        )

    def get_edge_weight(self, i, j, embed=None, filesuffix=None):
        if embed is None:
            if filesuffix is None:
                embed = self._Y
            else:
                embed = np.loadtxt('embedding_' + filesuffix + '.txt')
        if i == j:
            return 0
        else:
            S_hat = self.get_reconst_from_embed(embed[(i, j), :], filesuffix)
            return (S_hat[i, j] + S_hat[j, i]) / 2

    def get_reconstructed_adj(self, embed=None, node_l=None, filesuffix=None):
        if embed is None:
            if filesuffix is None:
                embed = self._Y
            else:
                embed = np.loadtxt('embedding_' + filesuffix + '.txt')
        S_hat = self.get_reconst_from_embed(embed, node_l, filesuffix)
        return graphify(S_hat)

    def get_reconst_from_embed(self, embed, node_l=None, filesuffix=None):
        if filesuffix is None:
            if node_l is not None:
                return self._decoder.predict(
                    embed,
                    batch_size=self._n_batch)[:, node_l]
            else:
                return self._decoder.predict(embed, batch_size=self._n_batch)
        else:
            try:
                decoder = model_from_json(
                    open('decoder_model_' + filesuffix + '.json').read()
                )
            except:
                print('Error reading file: {0}. Cannot load previous model'.format('decoder_model_'+filesuffix+'.json'))
                exit()
            try:
                decoder.load_weights('decoder_weights_' + filesuffix + '.hdf5')
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
    graph_util.print_graph_stats(G)
    t1 = time()
    embedding = SDNE(d=2, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3,
                     n_units=[50, 15], rho=0.3, n_iter=50, xeta=0.01,
                     n_batch=500,
                     modelfile=['./intermediate/enc_model.json',
                                './intermediate/dec_model.json'],
                     weightfile=['./intermediate/enc_weights.hdf5',
                                 './intermediate/dec_weights.hdf5'])
    embedding.learn_embedding(graph=G, edge_f=None,
                              is_weighted=True, no_python=True)
    print('SDNE:\n\tTraining time: %f' % (time() - t1))

    viz.plot_embedding2D(embedding.get_embedding(),
                         di_graph=G, node_colors=None)
    plt.show()
