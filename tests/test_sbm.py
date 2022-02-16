'''
Run the graph embedding methods on Karate graph and evaluate them on 
graph reconstruction and visualization. Please copy the 
gem/data/karate.edgelist to the working directory
'''
import os.path
import unittest

import matplotlib.pyplot as plt
from time import time
import networkx as nx
import pickle
import numpy as np

from gem.evaluation import visualize_embedding as viz

from gem.embedding.gf import GraphFactorization
from gem.embedding.hope import HOPE
from gem.embedding.lap import LaplacianEigenmaps
from gem.embedding.lle import LocallyLinearEmbedding
from gem.embedding.node2vec import node2vec
from gem.embedding.sdne import SDNE


from fit_model import fit_model


class SBMTest(unittest.TestCase):

    def setUp(self) -> None:
        # File that contains the edges. Format: source target
        # Optionally, you can add weights as third column: source target weight
        source_dir = os.path.dirname(os.path.abspath(__file__))
        file_prefix = os.path.join(source_dir, 'data/sbm.gpickle')

        # Load graph
        G = nx.read_gpickle(file_prefix)
        # convert G (networkx 1.x digraph) to networkx 2.x
        H = nx.DiGraph()
        H.add_nodes_from(G.node)
        for source_node in G.edge.keys():
            for target_node in G.edge[source_node].keys():
                H.add_edge(source_node, target_node)
        G = H
        try:
            node_colors = pickle.load(
                open(os.path.join(source_dir, 'data/sbm_node_labels.pickle'), 'rb')
            )
        except UnicodeDecodeError:
            node_colors = pickle.load(
                open(os.path.join(source_dir, 'data/sbm_node_labels.pickle'), 'rb'), encoding='latin1'
            )
        node_colors_arr = [None] * node_colors.shape[0]
        for idx in range(node_colors.shape[0]):
            node_colors_arr[idx] = np.where(node_colors[idx, :].toarray() == 1)[1][0]

        self.node_colors_arr = node_colors_arr
        self.G = G

    # todo: currently failing
    # def test_GraphFactorization(self):
    #    model = GraphFactorization(d=128, max_iter=1000, eta=1 * 10**-4, regu=1.0, data_set='sbm')
    #    target = np.loadtxt('smb_res/GraphFactorization.txt')
    #    self.internal_model_test(model, target)

    def test_HOPE(self):
        model = HOPE(d=256, beta=0.01)
        target = np.loadtxt('smb_res/HOPE.txt')
        self.internal_model_test(model, target)

    def test_LaplacianEigenmaps(self):
        model = LaplacianEigenmaps(d=128)
        target = np.loadtxt('smb_res/LaplacianEigenmaps.txt')
        self.internal_model_test(model, target)

    def test_LocallyLinearEmbedding(self):
        model = LocallyLinearEmbedding(d=128)
        target = np.loadtxt('smb_res/LocallyLinearEmbedding.txt')
        self.internal_model_test(model, target)

    # todo: currently failing
    # def test_node2vec(self):
    #    model = node2vec(d=182, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1, data_set='sbm')
    #    target = np.loadtxt('smb_res/node2vec.txt')
    #    self.internal_model_test(model, target)

    # todo: currently failing
    # def test_SDNE(self):
    #    model = SDNE(d=128, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3, n_units=[500, 300, ], rho=0.3, n_iter=30, xeta=0.001,
    #                 n_batch=500,
    #                 modelfile=['enc_model.json', 'dec_model.json'],
    #                 weightfile=['enc_weights.hdf5', 'dec_weights.hdf5'])
    #    target = np.loadtxt('smb_res/SDNE.txt')
    #    self.internal_model_test(model, target)

    def internal_model_test(self, model, target, verbose: bool = False):
        MAP, prec_curv, err, err_baseline = fit_model(self.G, model)
        # ---------------------------------------------------------------------------------
        if verbose:
            print(("\tMAP: {} \t preccision curve: {}\n\n\n\n" + '-' * 100).format(MAP, prec_curv[:5]))

        self.assertTrue(np.array_equal(model.get_embedding(), target))
