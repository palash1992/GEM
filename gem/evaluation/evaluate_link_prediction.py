from gem.evaluation import metrics
from gem.utils import evaluation_util, graph_util
import numpy as np
import networkx as nx
from utils import embed_util

def evaluateStaticLinkPrediction(digraph, graph_embedding, 
                                 train_ratio=0.8,
                                 n_sample_nodes=None,
                                 sample_ratio_e=None,
                                 no_python=False,
                                 is_undirected=True):
    node_num = digraph.number_of_nodes()
    # seperate train and test graph
    train_digraph, test_digraph = evaluation_util.splitDiGraphToTrainTest(digraph,
                                                                          train_ratio=train_ratio,
                                                                          is_undirected=is_undirected)
    if not nx.is_connected(train_digraph.to_undirected()):
      train_digraph = max(nx.weakly_connected_component_subgraphs(train_digraph),key=len)
      tdl_nodes = train_digraph.nodes()
      nodeListMap = dict(zip(tdl_nodes, range(len(tdl_nodes))))
      nx.relabel_nodes(train_digraph, nodeListMap, copy=False)
      test_digraph = test_digraph.subgraph(tdl_nodes)
      nx.relabel_nodes(test_digraph, nodeListMap, copy=False)
    # learning graph embedding
    X, _ = graph_embedding.learn_embedding(graph=train_digraph, no_python=no_python)
    node_l = None
    if n_sample_nodes:
        test_digraph, node_l = graph_util.sample_graph(test_digraph, n_sample_nodes)
        X = X[node_l]

    # evaluation
    if sample_ratio_e:
        eval_edge_pairs = evaluation_util.getRandomEdgePairs(node_num, sample_ratio_e, is_undirected)
    else:
        eval_edge_pairs = None
    estimated_adj = graph_embedding.get_reconstructed_adj(X, node_l)
    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(estimated_adj, 
        is_undirected=is_undirected, edge_pairs=eval_edge_pairs)

    filtered_edge_list = [e for e in predicted_edge_list if not train_digraph.has_edge(e[0], e[1])]

    MAP = metrics.computeMAP(filtered_edge_list, test_digraph)
    prec_curv, _ = metrics.computePrecisionCurve(filtered_edge_list, test_digraph)
    return (MAP, prec_curv)