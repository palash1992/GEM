from gem.evaluation import evaluate_graph_reconstruction as gr


def fit_model(graph, model, verbose: bool = False):
    if verbose:
        print('Num nodes: %d, num edges: %d' % (graph.number_of_nodes(), graph.number_of_edges()))
    # Learn embedding - accepts a networkx graph or file with edge list
    y = model.learn_embedding(graph=graph, edge_f=None, is_weighted=True, no_python=True)
    # Evaluate on graph reconstruction
    return gr.evaluateStaticGraphReconstruction(graph, model, y, None)
