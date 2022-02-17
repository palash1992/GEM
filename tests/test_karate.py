'''
Run the graph embedding methods on Karate graph and evaluate them on 
graph reconstruction and visualization. Please copy the 
gem/data/karate.edgelist to the working directory
'''
import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

from gem.utils import graph_util
from gem.evaluation import visualize_embedding as viz

from gem.embedding.gf import GraphFactorization
from gem.embedding.hope import HOPE
from gem.embedding.lap import LaplacianEigenmaps
from gem.embedding.lle import LocallyLinearEmbedding
from gem.embedding.node2vec import node2vec
from gem.embedding.sdne import SDNE

from tests.fit_model import fit_model


class KarateTest(unittest.TestCase):

    def setUp(self) -> None:
        # File that contains the edges. Format: source target
        # Optionally, you can add weights as third column: source target weight
        self.source_dir = os.path.dirname(os.path.abspath(__file__))
        edge_f = os.path.join(self.source_dir, 'data/karate.edgelist')
        # Specify whether the edges are directed
        isDirected = True

        # Load graph
        G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=isDirected)
        self.G = G.to_directed()

    def test_GraphFactorization(self):
        model = GraphFactorization(d=2, max_iter=50000, eta=1 * 10**-4, regu=1.0, data_set='karate')
        target = np.loadtxt(os.path.join(self.source_dir, 'karate_res/GraphFactorization.txt'))
        self.internal_model_test(model, target, mae_close=True)

    def test_HOPE(self):
        model = HOPE(d=4, beta=0.01)
        target = np.loadtxt(os.path.join(self.source_dir, 'karate_res/HOPE.txt'))
        self.internal_model_test(model, target)

    def test_LaplacianEigenmaps(self):
        model = LaplacianEigenmaps(d=2)
        target = np.loadtxt(os.path.join(self.source_dir, 'karate_res/LaplacianEigenmaps.txt'))
        self.internal_model_test(model, target)

    def test_LocallyLinearEmbedding(self):
        model = LocallyLinearEmbedding(d=2)
        target = np.loadtxt(os.path.join(self.source_dir, 'karate_res/LocallyLinearEmbedding.txt'))
        self.internal_model_test(model, target, mae_close=True)

    def test_node2vec(self):
        model = node2vec(d=2, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1)
        target = np.loadtxt('karate_res/node2vec.txt')
        self.internal_model_test(model, target, mae_close=True)

    def test_SDNE(self):
        model = SDNE(d=2, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3,n_units=[50, 15,], rho=0.3, n_iter=50,
                     xeta=0.01,n_batch=100, modelfile=['enc_model.json', 'dec_model.json'],
                     weightfile=['enc_weights.hdf5', 'dec_weights.hdf5'])
        target = np.loadtxt(os.path.join(self.source_dir,'karate_res/SDNE.txt'))
        self.internal_model_test(model, target, mae_close=True)

    def internal_model_test(self, model, target, verbose: bool = False, mae_close: bool = False):
        MAP, prec_curv, err, err_baseline = fit_model(self.G, model)
        # ---------------------------------------------------------------------------------
        if verbose:
            print(("\tMAP: {} \t preccision curve: {}\n\n\n\n" + '-' * 100).format(MAP, prec_curv[:5]))
        # ---------------------------------------------------------------------------------
        if not mae_close:
            self.assertTrue(np.allclose(model.get_embedding(), target))
        else:
            self.assertTrue(abs(np.mean(target-model.get_embedding())) < .3)

