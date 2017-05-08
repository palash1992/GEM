import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

def plot_embedding2D(node_pos, node_colors=None, di_graph=None):
    node_num, embedding_dimension = node_pos.shape
    if(embedding_dimension > 2):
        print "Embedding dimensiion greater than 2, use tSNE to reduce it to 2"
        model = TSNE(n_components=2)
        node_pos = model.fit_transform(node_pos)

    if di_graph is None:
        # plot using plt scatter
        plt.scatter(node_pos[:,0], node_pos[:,1], c=node_colors)
    else:
        # plot using networkx with edge structure
        pos = {}
        for i in xrange(node_num):
            pos[i] = node_pos[i, :]
        if node_colors:
            nx.draw_networkx_nodes(di_graph, pos, node_color=node_colors, width=0.1, node_size=100, arrows=False, alpha=0.8, font_size=5)
        else:
            nx.draw_networkx(di_graph, pos, node_color=node_colors, width=0.1, node_size=300, arrows=False, alpha=0.8, font_size=12)
