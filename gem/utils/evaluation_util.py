import numpy as np
import secrets


def get_random_edge_pairs(node_num, sample_ratio=0.01, is_undirected=True):
    num_pairs = int(sample_ratio * node_num * (node_num - 1))
    if is_undirected:
        num_pairs = num_pairs / 2
    current_sets = set()
    while len(current_sets) < num_pairs:
        p = (secrets.randbelow(node_num), secrets.randbelow(node_num))
        if p in current_sets:
            continue
        if is_undirected and (p[1], p[0]) in current_sets:
            continue
        current_sets.add(p)
    return list(current_sets)


def get_edge_list_from_adj_mtrx(adj, threshold=0.0, is_undirected=True, edge_pairs=None):
    result = []
    node_num = adj.shape[0]
    if edge_pairs:
        for (st, ed) in edge_pairs:
            if adj[st, ed] >= threshold:
                result.append((st, ed, adj[st, ed]))
    else:
        for i in range(node_num):
            for j in range(node_num):
                if j == i:
                    continue
                if is_undirected and i >= j:
                    continue
                if adj[i, j] > threshold:
                    result.append((i, j, adj[i, j]))
    return result


def split_di_graph_to_train_test(di_graph, train_ratio, is_undirected=True):
    train_digraph = di_graph.copy()
    test_digraph = di_graph.copy()
    for (st, ed, w) in list(di_graph.edges(data='weight', default=1)):
        if is_undirected and st >= ed:
            continue
        if np.random.uniform() <= train_ratio:
            test_digraph.remove_edge(st, ed)
            if is_undirected:
                test_digraph.remove_edge(ed, st)
        else:
            train_digraph.remove_edge(st, ed)
            if is_undirected:
                train_digraph.remove_edge(ed, st) 
    return train_digraph, test_digraph
