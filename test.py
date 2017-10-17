from gem.evaluation import evaluate_graph_reconstruction as gr
from gem.utils import graph_util

from gem.embedding.sdne import SDNE as sdne
from gem.embedding.gf import GraphFactorization as gf
from gem.embedding.hope import HOPE as hope


# Instatiate the embedding methods with hyperparameters
models = []
models.append((hope(2, 1),"Hope"))
models.append((gf(2, 100000, 1*10**-4, 1.0),"GraphFactorization"))

#models.append((sdne(128,0.1,0.1,0.1,0.1,3,128,0.1,0.1,5000,50),"SDNE"))



for model,name in models:
    # Load graph
    graph  = graph_util.loadGraphFromEdgeListTxt('gem/data/karate.edgelist')
    
    # Learn embedding - accepts a networkx graph or file with edge list
    Y,  t  = model.learn_embedding(graph,  edge_f=None, is_weighted=True, no_python=True)
    
    # Evaluate on graph reconstruction
    MAP, prec_curv = gr.evaluateStaticGraphReconstruction(graph, model, Y, None)

    print("-----------------------------------------------------------------------------\n\
            Model: {} \n MAP: {}".format(name,MAP))
