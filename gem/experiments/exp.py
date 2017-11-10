try: import cPickle as pickle
except: import pickle
from time import time
from argparse import ArgumentParser
import importlib
import json
import cPickle
import networkx as nx
import itertools
import pdb
import sys
sys.path.insert(0, './')

from gem.utils      import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation.evaluate_graph_reconstruction import expGR
from gem.evaluation.evaluate_link_prediction import expLP
from gem.evaluation.evaluate_node_classification import expNC
from gem.evaluation.visualize_embedding import expVis

methClassMap = {"gf": "GraphFactorization",
                "hope": "HOPE",
                "lap": "LaplacianEigenmaps",
                "lle": "LocallyLinearEmbedding",
                "node2vec": "node2vec",
                "sdne": "SDNE"}


def learn_emb(MethObj, di_graph, params, res_pre, m_summ):
    if params["experiments"] == ["lp"]:
        X = None
    else:
        print 'Learning Embedding: %s' % m_summ
        if not bool(int(params["load_emb"])):
            X, learn_t = MethObj.learn_embedding(graph=di_graph,
                                                 edge_f=None,
                                                 no_python=True)
            print '\tTime to learn embedding: %f sec' % learn_t
            pickle.dump(X, open('%s_%s.emb' % (res_pre, m_summ), 'wb'))
            pickle.dump(learn_t,
                        open('%s_%s.learnT' % (res_pre, m_summ), 'wb'))
        else:
            X = pickle.load(open('%s_%s.emb' % (res_pre, m_summ),
                                 'rb'))
            try:
                learn_t = pickle.load(open('%s_%s.learnT' % (res_pre, m_summ),
                                           'rb'))
                print '\tTime to learn emb.: %f sec' % learn_t
            except IOError:
                print '\tTime info not found'
    return X


def run_exps(MethObj, di_graph, data_set, node_labels, params):
    m_summ = MethObj.get_method_summary()
    res_pre = "gem/results/%s" % data_set
    X = learn_emb(MethObj, di_graph, params, res_pre, m_summ)
    if "gr" in params["experiments"]:
        expGR(di_graph, MethObj,
              X, params["n_sample_nodes"],
              params["rounds"], res_pre,
              m_summ, is_undirected=params["is_undirected"])
    if "lp" in params["experiments"]:
        expLP(di_graph, MethObj,
              params["n_sample_nodes"],
              params["rounds"], res_pre,
              m_summ, is_undirected=params["is_undirected"])
    if "nc" in params["experiments"]:
        if "nc_test_ratio_arr" not in params:
            print('NC test ratio not provided')
        else:
            expNC(X, node_labels, params["nc_test_ratio_arr"],
                  params["rounds"], res_pre,
                  m_summ)
    if "viz" in params["experiments"]:
        if MethObj.get_method_name() == 'hope_gsvd':
            d = X.shape[1] / 2
            expVis(X[:, :d], res_pre, m_summ,
                   node_labels=node_labels, di_graph=di_graph)
        else:
            expVis(X, res_pre, m_summ,
                   node_labels=node_labels, di_graph=di_graph)


def call_exps(params, data_set):
    print('Dataset: %s' % data_set)
    model_hyp = json.load(
        open('gem/experiments/config/%s.conf' % data_set, 'r')
    )
    if bool(params["node_labels"]):
        node_labels = cPickle.load(
            open('gem/data/%s/node_labels.pickle' % data_set, 'rb')
        )
    else:
        node_labels = None
    di_graph = nx.read_gpickle('gem/data/%s/graph.gpickle' % data_set)
    for d, meth in itertools.product(params["dimensions"], params["methods"]):
        dim = int(d)
        MethClass = getattr(
            importlib.import_module("gem.embedding.%s" % meth),
            methClassMap[meth]
        )
        hyp = {"d": dim}
        hyp.update(model_hyp[meth])
        MethObj = MethClass(hyp)
        run_exps(MethObj, di_graph, data_set, node_labels, params)


if __name__ == '__main__':
    ''' Sample usage
    python experiments/exp.py -data sbm -dim 128 -meth sdne -exp gr,lp
    '''
    t1 = time()
    parser = ArgumentParser(description='Graph Embedding Experiments')
    parser.add_argument('-data', '--data_sets',
                        help='dataset names (default: sbm)')
    parser.add_argument('-dim', '--dimensions',
                        help='embedding dimensions list(default: 2^1 to 2^8)')
    parser.add_argument('-meth', '--methods',
                        help='method list (default: all methods)')
    parser.add_argument('-exp', '--experiments',
                        help='exp list (default: gr,lp,viz,nc)')
    parser.add_argument('-lemb', '--load_emb',
                        help='load saved embeddings (default: False)')
    parser.add_argument('-lexp', '--load_exp',
                        help='load saved experiment results (default: False)')
    parser.add_argument('-rounds', '--rounds',
                        help='number of rounds (default: 5)')
    parser.add_argument('-plot', '--plot',
                        help='plot the results (default: False)')
    parser.add_argument('-saveMAP', '--save_MAP',
                        help='save MAP in a latex table (default: False)')

    params = json.load(open('gem/experiments/config/params.conf', 'r'))
    args = vars(parser.parse_args())
    for k, v in args.iteritems():
        if v is not None:
            params[k] = v
    params["experiments"] = params["experiments"].split(',')
    params["data_sets"] = params["data_sets"].split(',')
    params["rounds"] = int(params["rounds"])
    params["n_sample_nodes"] = int(params["n_sample_nodes"])
    params["is_undirected"] = bool(int(params["is_undirected"]))
    if params["methods"] == "all":
        params["methods"] = methClassMap.keys()
    else:
        params["methods"] = params["methods"].split(',')
    params["dimensions"] = params["dimensions"].split(',')
    if "nc_test_ratio_arr" in params:
        params["nc_test_ratio_arr"] = params["nc_test_ratio_arr"].split(',')
        params["nc_test_ratio_arr"] = \
            [float(ratio) for ratio in params["nc_test_ratio_arr"]]
    for data_set in params["data_sets"]:
        call_exps(params, data_set)
