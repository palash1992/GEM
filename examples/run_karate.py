'''
Run the graph embedding methods on Karate graph and evaluate them on 
graph reconstruction and visualization. Please copy the 
gem/data/karate.edgelist to the working directory
'''
import matplotlib.pyplot as plt
from time import time

from gem.utils      import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation import evaluate_graph_reconstruction as gr

from gem.embedding.gf       import GraphFactorization
from gem.embedding.hope     import HOPE
from gem.embedding.lap      import LaplacianEigenmaps
from gem.embedding.lle      import LocallyLinearEmbedding
from gem.embedding.node2vec import node2vec
from gem.embedding.sdne     import SDNE
from argparse import ArgumentParser


if __name__ == '__main__':
    ''' Sample usage
    python run_karate.py -node2vec 1
    '''
    parser = ArgumentParser(description='Graph Embedding Experiments on Karate Graph')
    parser.add_argument('-node2vec', '--node2vec',
                        help='whether to run node2vec (default: False)')
    args = vars(parser.parse_args())
    try:
        run_n2v = bool(int(args["node2vec"]))
    except:
        run_n2v = False

    # File that contains the edges. Format: source target
    # Optionally, you can add weights as third column: source target weight
    edge_f = 'data/karate.edgelist'
    # Specify whether the edges are directed
    isDirected = True

    # Load graph
    G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=isDirected)
    G = G.to_directed()

    models = []
    # Load the models you want to run
    models.append(GraphFactorization(d=2, max_iter=50000, eta=1 * 10**-4, regu=1.0))
    models.append(HOPE(d=4, beta=0.01))
    models.append(LaplacianEigenmaps(d=2))
    models.append(LocallyLinearEmbedding(d=2))
    if run_n2v:
        models.append(
            node2vec(d=2, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1)
        )
    models.append(SDNE(d=2, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3,n_units=[50, 15,], rho=0.3, n_iter=50, xeta=0.01,n_batch=100,
                    modelfile=['enc_model.json', 'dec_model.json'],
                    weightfile=['enc_weights.hdf5', 'dec_weights.hdf5']))

    # For each model, learn the embedding and evaluate on graph reconstruction and visualization
    for embedding in models:
        print ('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
        t1 = time()
        # Learn embedding - accepts a networkx graph or file with edge list
        Y, t = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
        print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
        # Evaluate on graph reconstruction
        MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, embedding, Y, None)
        #---------------------------------------------------------------------------------
        print(("\tMAP: {} \t preccision curve: {}\n\n\n\n"+'-'*100).format(MAP,prec_curv[:5]))
        #---------------------------------------------------------------------------------
        # Visualize
        viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=None)
        plt.show()
        plt.clf()
