# GEM: Graph Embedding Methods
GEM is a Python module that implements many graph (a.k.a. network) embedding algorithms. GEM is distributed under BSD license.

The module was developed and is maintained by Palash Goyal.

## Implemented Methods
GEM implements the following graph embedding techniques:
* [Laplacian Eigenmaps](http://yeolab.weebly.com/uploads/2/5/5/0/25509700/belkin_laplacian_2003.pdf)
* [Locally Linear Embedding](http://www.robots.ox.ac.uk/~az/lectures/ml/lle.pdf)
* [Graph Factorization](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40839.pdf)
* [Higher-Order Proximity preserved Embedding (HOPE)](http://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf)
* [Structural Deep Network Embedding (SDNE)](http://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf)
* [node2vec](http://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf)

A survey of these methods can be found in [Graph Embedding Techniques, Applications, and Performance: A Survey](https://arxiv.org/abs/1705.02801).

## Graph Format
We store all graphs using the [DiGraph](http://networkx.readthedocs.io/en/networkx-1.11/reference/classes.digraph.html) as **directed weighted graph** in python package networkx. The weight of an edge is stored as attribute "weight". We save each edge in undirected graph as two directed edges.

The graphs are saved using `nx.write_gpickle` in the networkx format and can be loaded by using `nx.read_gpickle`.

## Repository Structure
* **gem/embedding**: existing approaches for graph embedding, where each method is a separate file
* **gem/evaluation**: evaluation tasks for graph embedding, including graph reconstruction, link prediction, node classification and visualization
* **gem/utils**: utility functions for graph manipulation, evaluation and etc.
* **gem/data**: input test graph (currently has [Zachary's Karate graph](https://en.wikipedia.org/wiki/Zachary%27s_karate_club))
* **gem/c_src**: source files for methods implemented in C++
* **gem/c_ext**: Python interface for source files in c_src using [Boost.Python](http://www.boost.org/doc/libs/1_64_0/libs/python/doc/html/index.html)

## Dependencies
GEM is tested to work on Python 2.7 and Python 3.6

The required dependencies are: Numpy >= 1.12.0, SciPy >= 0.19.0, Networkx >= 1.11, Scikit-learn >= 0.18.1.

To run SDNE, GEM requires Theano >= 0.9.0 and Keras = 2.0.2.

In case of Python 3, make sure it was compiled with `./configure --enable-shared`, and that you have `/usr/local/bin/python` in your `LD_LIBRARY_PATH`.

## Install
The package uses setuptools, which is a common way of installing python modules. To install in your home directory, use:
```bash
    python setup.py install --user
```

To install for all users on Unix/Linux:
```bash 
    sudo python setup.py install
```

You also can use `python3` instead of `python`.

To install node2vec as part of the package, recompile from https://github.com/snap-stanford/snap and add node2vec executable to system path.
To grant executable permission, run: chmod +x node2vec

## Usage
Run the methods on Karate graph and evaluate them on graph reconstruction and visualization:

```python
import matplotlib.pyplot as plt

from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation import evaluate_graph_reconstruction as gr
from time import time

from gem.embedding.gf       import GraphFactorization
from gem.embedding.hope     import HOPE
from gem.embedding.lap      import LaplacianEigenmaps
from gem.embedding.lle      import LocallyLinearEmbedding
from gem.embedding.node2vec import node2vec
from gem.embedding.sdne     import SDNE

# File that contains the edges. Format: source target
# Optionally, you can add weights as third column: source target weight
# Copy the gem/data/karate.edgelist to the working directory
edge_f = 'karate.edgelist'
# Specify whether the edges are directed
isDirected = True

# Load graph
G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=isDirected)
G = G.to_directed()

models = []
# You can comment out the methods you don't want to run
models.append(GraphFactorization(d=2, max_iter=100000, eta=1*10**-4, regu=1.0))
models.append(HOPE(d=4, beta=0.01))
models.append(LaplacianEigenmaps(d=2))
models.append(LocallyLinearEmbedding(d=2))
models.append(node2vec(d=2, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1))
models.append(SDNE(d=2, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3,n_units=[50, 15,], rho=0.3, n_iter=50, xeta=0.01,n_batch=500,
                modelfile=['./intermediate/enc_model.json', './intermediate/dec_model.json'],
                weightfile=['./intermediate/enc_weights.hdf5', './intermediate/dec_weights.hdf5']))

for embedding in models:
    print ('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
    t1 = time()
    # Learn embedding - accepts a networkx graph or file with edge list
    Y, t = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
    print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
    # Evaluate on graph reconstruction
    MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, embedding, Y, None)
    # Visualize
    viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=None)
    plt.show()
```


## Cite
    @article{goyal2017graph,
        title = "Graph embedding techniques, applications, and performance: A survey",
        journal = "Knowledge-Based Systems",
        year = "2018",
        issn = "0950-7051",
        doi = "https://doi.org/10.1016/j.knosys.2018.03.022",
        url = "http://www.sciencedirect.com/science/article/pii/S0950705118301540",
        author = "Palash Goyal and Emilio Ferrara",
        keywords = "Graph embedding techniques, Graph embedding applications, Python graph embedding methods GEM library"
    }


