import numpy as np

precision_pos = [2, 10, 100, 200, 300, 500, 1000]


def computePrecisionCurve(predicted_edge_list, true_digraph, max_k=-1):
    if max_k == -1:
        max_k = len(predicted_edge_list)
    else:
        max_k = min(max_k, len(predicted_edge_list))

    sorted_edges = sorted(predicted_edge_list, key=lambda x: x[2], reverse=True)

    precision_scores = []
    delta_factors = []
    correct_edge = 0
    for i in range(max_k):
        if true_digraph.has_edge(sorted_edges[i][0], sorted_edges[i][1]):
            correct_edge += 1
            delta_factors.append(1.0)
        else:
            delta_factors.append(0.0)
        precision_scores.append(1.0 * correct_edge / (i + 1))
    return precision_scores, delta_factors


def computeMAP(predicted_edge_list, true_digraph, max_k=-1, is_undirected=False):
    node_num = len(true_digraph.nodes)
    node_edges = []
    for i in range(node_num):
        node_edges.append([])
    for (st, ed, w) in predicted_edge_list:
        node_edges[st].append((st, ed, w))
    node_ap = [0.0] * node_num
    count = 0
    for i in range(node_num):
        if not is_undirected and true_digraph.out_degree(i) == 0:
            continue
        count += 1
        precision_scores, delta_factors = computePrecisionCurve(node_edges[i], true_digraph, max_k)
        precision_rectified = [p * d for p, d in zip(precision_scores, delta_factors)]
        if sum(delta_factors) == 0:
            node_ap[i] = 0
        else:
            node_ap[i] = float(sum(precision_rectified) / sum(delta_factors))
    return sum(node_ap) / count
