---
title: 'GEM: A Python package for graph embedding methods'
tags:
  - Python
  - networks
  - graphs
  - embedding
  - graph embedding
  - network analysis
  - graph visualization
authors:
  - name: Palash Goyal
    orcid: 0000-0003-2455-2160
    affiliation: 1
  - name: Emilio Ferrara
    orcid: 0000-0002-1942-2831
    affiliation: 1
affiliations:
 - name: USC Information Sciences Institute
   index: 1
date: 08 July 2018
bibliography: paper.bib
---

# Summary

Many physical systems in the world involve interactions between different entities and can be represented as graphs. Understanding the structure and analyzing properties of graphs are hence paramount to developing insights into the physical systems. Graph embedding, which aims to represent a graph in a low dimensional vector space, takes a step in this direction. The embeddings can be used for various tasks on graphs such as visualization, clustering, classification and prediction.

``GEM`` is a Python package which offers a general framework for graph embedding methods. It implements many state-of-the-art embedding techniques including Locally Linear Embedding [@roweis2000nonlinear], Laplacian Eigenmaps [@belkin2002laplacian], Graph Factorization [@ahmed2013distributed], HOPE [@ou2016asymmetric], SDNE [@wang2016structural] and node2vec [@grover2016node2vec]. It is formatted such that new methods can be easily added for comparison. Furthermore, the framework implements several functions to evaluate the quality of obtained embedding including graph reconstruction, link prediction, visualization and node classification. It supports many edge reconstruction metrics including cosine similarity, euclidean distance and decoder based. For node classification, it defaults to one-vs-rest logistic regression classifier and supports other classifiers. For faster execution, C++ backend is integrated using Boost for supported methods.

``GEM`` was designed to be used by researchers studying graphs. It has already been used in a number of scientific publications to compare novel methods against the state-of-the-art and general evaluation [@salehi2017properties, @lyu2017enhancing]. A paper showcasing the results using ``GEM`` on various real world datasets can be accessed [@goyal2017graph]. The source code of ``GEM`` is made available at <https://github.com/palash1992/GEM>. Bug reports and feedback can be directed to the Github issues page (<https://github.com/palash1992/GEM/issues>).


# References
