from gem.evaluation import metrics
from gem.utils import evaluation_util, graph_util

def evaluateStaticGraphReconstruction(digraph, graph_embedding,
                                      X_stat, node_l=None, file_suffix=None,
                                      sample_ratio_e=None, is_undirected=True):
    node_num = digraph.number_of_nodes()
    # evaluation
    if sample_ratio_e:
        eval_edge_pairs = evaluation_util.getRandomEdgePairs(node_num, sample_ratio_e, is_undirected)
    else:
        eval_edge_pairs = None
    if file_suffix is None:
        estimated_adj = graph_embedding.get_reconstructed_adj(X_stat, node_l)
    else:
        estimated_adj = graph_embedding.get_reconstructed_adj(X_stat, file_suffix, node_l)
    
    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(estimated_adj, 
        is_undirected=is_undirected, edge_pairs=eval_edge_pairs)
    MAP = metrics.computeMAP(predicted_edge_list, digraph)
    prec_curv, _ = metrics.computePrecisionCurve(predicted_edge_list, digraph)

    return (MAP, prec_curv)
