[![Python application](https://github.com/jernsting/GEM/actions/workflows/lint_and_test.yml/badge.svg)](https://github.com/jernsting/GEM/actions/workflows/lint_and_test.yml)
[![Coverage Status](https://coveralls.io/repos/github/jernsting/GEM/badge.svg?branch=master)](https://coveralls.io/github/jernsting/GEM?branch=master)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=jernsting_GEM&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=jernsting_GEM)

# GEM: Graph Embedding Methods
Many physical systems in the world involve interactions between different entities and can be represented as graphs. Understanding the structure and analyzing properties of graphs are hence paramount to developing insights into the physical systems. Graph embedding, which aims to represent a graph in a low dimensional vector space, takes a step in this direction. The embeddings can be used for various tasks on graphs such as visualization, clustering, classification and prediction.

``GEM`` is a Python package which offers a general framework for graph embedding methods. It implements many state-of-the-art embedding techniques including [Locally Linear Embedding](http://www.robots.ox.ac.uk/~az/lectures/ml/lle.pdf), [Laplacian Eigenmaps](http://yeolab.weebly.com/uploads/2/5/5/0/25509700/belkin_laplacian_2003.pdf), [Graph Factorization](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40839.pdf), [Higher-Order Proximity preserved Embedding (HOPE)](http://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf), [Structural Deep Network Embedding (SDNE)](http://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf) and [node2vec](http://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf). It is formatted such that new methods can be easily added for comparison. Furthermore, the framework implements several functions to evaluate the quality of obtained embedding including graph reconstruction, link prediction, visualization and node classification. It supports many edge reconstruction metrics including cosine similarity, euclidean distance and decoder based. For node classification, it defaults to one-vs-rest logistic regression classifier and supports other classifiers. For faster execution, C++ backend is integrated using Boost for supported methods. A paper showcasing the results using ``GEM`` on various real world datasets can be accessed through [Graph Embedding Techniques, Applications, and Performance: A Survey](https://arxiv.org/abs/1705.02801). The library is also published as [GEM: A Python package for graph embedding methods](https://doi.org/10.21105/joss.00876).

Please refer [https://palash1992.github.io/GEM/](https://palash1992.github.io/GEM/) to access the readme as a webpage.

**Update**: Note that this is a library for static graph embedding methods. For evolving graph embedding methods, please refer [DynamicGEM](https://github.com/palash1992/DynamicGEM). We also recently released Youtube dynamic graph data set which can be found at [YoutubeGraph-Dyn](https://github.com/palash1992/YoutubeGraph-Dyn).

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
nxt_gem is tested to work on Python 3.9

The required dependencies are: Numpy >= 1.12.0, SciPy >= 0.19.0, Networkx >= 2.4, Scikit-learn >= 0.18.1.

To run SDNE, GEM requires Theano >= 0.9.0 and tensorflow.

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

Or installing via pipwith git: 
```bash 
pip install git+https://github.com/jernsting/nxt_gem.git
```

## Usage
See examples.

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
    @article{goyal3gem,
      title={GEM: A Python package for graph embedding methods},
      author={Goyal, Palash and Ferrara, Emilio},
      journal={Journal of Open Source Software},
      volume={3},
      number={29},
      pages={876}
    }

