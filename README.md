# GEM
GEM is a Python module that implements many graph (a.k.a. network) embedding algorithms. GEM is distributed under BSD license.

The module was developed and is maintained by Palash Goyal.


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
GEM is tested to work on Python 2.7.

The required dependencies are: Numpy >= 1.12.0, SciPy >= 0.19.0, Networkx >= 1.11, Scikit-learn >= 0.18.1.

To run SDNE, GEM requires Theano >= 0.9.0 and Keras = 2.0.2.

## Install
The package uses setuptools, which is a common way of installing python modules. To install in your home directory, use:

    python setup.py install --user

To install for all users on Unix/Linux:
    
    sudo python setup.py install

## Usage
Run Graph Factorization on Karate graph and evaluate it on graph reconstruction:

    from gem.embedding.gf import GraphFactorization as gf
    from gem.evaluation import evaluate_graph_reconstruction as gr
    grom gem.utils import graph_util
    # Instatiate the embedding method with hyperparameters
    em = gf(2, 100000, 1*10**-4, 1.0)
    # Load graph
    graph = graph_util.loadGraph('gem/data/karate.edgelist')
    # Learn embedding - accepts a networkx graph or file with edge list
    Y, t = em.learn_embedding(graph, edge_f=None, is_weighted=True, no_python=True)
    # Evaluate on graph reconstruction
    MAP, prec_curv = gr.evaluateStaticGraphReconstruction(graph, em, Y, None)


