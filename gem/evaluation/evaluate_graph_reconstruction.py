import pickle
from gem.evaluation import metrics
from gem.utils import evaluation_util, graph_util
import networkx as nx
import numpy as np


def evaluateStaticGraphReconstruction(digraph, graph_embedding,
                                      X_stat, node_l=None, file_suffix=None,
                                      sample_ratio_e=None, is_undirected=True,
                                      is_weighted=False):
    node_num = len(digraph.nodes)
    # evaluation
    if sample_ratio_e:
        eval_edge_pairs = evaluation_util.get_random_edge_pairs(
            node_num,
            sample_ratio_e,
            is_undirected
        )
    else:
        eval_edge_pairs = None
    if file_suffix is None:
        estimated_adj = graph_embedding.get_reconstructed_adj(X_stat, node_l)
    else:
        estimated_adj = graph_embedding.get_reconstructed_adj(
            X_stat,
            file_suffix,
            node_l
        )
    predicted_edge_list = evaluation_util.get_edge_list_from_adj_mtrx(
        estimated_adj,
        is_undirected=is_undirected,
        edge_pairs=eval_edge_pairs
    )
    MAP = metrics.computeMAP(predicted_edge_list, digraph, is_undirected=is_undirected)
    prec_curv, _ = metrics.computePrecisionCurve(predicted_edge_list, digraph)
    # If weighted, compute the error in reconstructed weights of observed edges
    if is_weighted:
        digraph_adj = nx.to_numpy_matrix(digraph)
        estimated_adj[digraph_adj == 0] = 0
        err = np.linalg.norm(digraph_adj - estimated_adj)
        err_baseline = np.linalg.norm(digraph_adj)
    else:
        err = None
        err_baseline = None
    return MAP, prec_curv, err, err_baseline
