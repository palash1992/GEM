#include <boost/python/numpy.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <fstream>
#include <iomanip>
#include <string>
#include <numeric>
#include <algorithm> 
#include <unistd.h>
#include <cmath>
#include <chrono>
#include <random>
#include <iostream>
#include <assert.h>
#include <vector>

namespace p = boost::python;
namespace np = boost::python::numpy;

int n_nodes;
int n_edges;
int **edges;
float *edge_weights;
float **X;

void gen_rand_network(int n, int m)
{
    int row;
    n_nodes = n;
    n_edges = m;
    edges = (int **)malloc(n_edges*sizeof(int *));
    for(int i = 0; i < n_edges; i++)
        edges[i] = (int *)malloc(2*sizeof(int));

    edge_weights = (float *)malloc(n_edges*sizeof(float));
    assert(edges != NULL);
    row = 0;
    while(row < n_edges)
    {
        edges[row][0] = rand()%n_nodes;
        edges[row][1] = rand()%n_nodes;
        edge_weights[row] = 1.0;
        row++;
    }
}

void init_embedding(int d)
{
    X = (float **)malloc(n_nodes*sizeof(float *));
    for(int i = 0; i < n_nodes; i++)
        X[i] = (float *)malloc(d*sizeof(float));
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<float> distribution (0.0,1.0);
    for (int i = 0; i < n_nodes; i++)
        for(int j = 0; j < d; j++)
            X[i][j] = 0.01*distribution(generator);
}

void load_network(std::string f_name, bool is_weighted)
{
    std::ifstream f;
    int v_i, v_j, row;
    float w;

    f.open(f_name);
    f >> n_nodes;
    f >> n_edges;
    edges = (int **)malloc(n_edges*sizeof(int *));
    for(int i = 0; i < n_edges; i++)
        edges[i] = (int *)malloc(2*sizeof(int));

    edge_weights = (float *)malloc(n_edges*sizeof(float));
    assert(edges != NULL);

    row = 0;
    if(is_weighted)
    {
        while(f >> v_i  >> v_j >>  w)
        {
            edges[row][0] = v_i;
            edges[row][1] = v_j;
            edge_weights[row] = w;
            row++;
        }
    }
    else
    {
        while(f >> v_i  >> v_j)
        {
            edges[row][0] = v_i;
            edges[row][1] = v_j;
            edge_weights[row] = 1.0;
            row++;
        }
    }
    f.close();
}

void _print_f_value(int d)
{
    float f1 = 0.0, f2 = 0.0;
    float i_j_dot, w_ij;
    int v_i, v_j;
    for(int edge_id = 0; edge_id < n_edges; edge_id++)
    {
        v_i = edges[edge_id][0];
        v_j = edges[edge_id][1];
        w_ij = edge_weights[edge_id];
        i_j_dot = 0.0;
        for(int d_id = 0; d_id < d; d_id++)
            i_j_dot += X[v_i][d_id]*X[v_j][d_id];
        f1 += (w_ij - i_j_dot)*(w_ij - i_j_dot);
    }
    for(int node_id = 0; node_id < n_nodes; node_id++)
        for(int d_id = 0; d_id < d; d_id++)
            f2 += X[node_id][d_id]*X[node_id][d_id];
    std::cout << "\t\tObjective: " << f1+f2 << ", f1: " << f1 << ", f2:" << f2 << std::endl;
}

void saveEmbToTxt(std::string of_name, int d)
{
    std::ofstream f;
    f.open(of_name);
    f << n_nodes << " " << d << std::endl;
    for(int node_id = 0; node_id < n_nodes; node_id++)
    {
        f << node_id;
        for(int d_id = 0; d_id < d; d_id++)
            f << " " << X[node_id][d_id];
        f << std::endl;
    }
    f.close();

}
void learn_embedding(std::string if_name, std::string of_name, bool verbose, bool is_weighted, int d, float eta, float regu, int max_iter)
{
    clock_t t;
    srand(time(NULL));
    t = clock();
    load_network(if_name, is_weighted);
    // gen_rand_network(1e3, 3e4);
    init_embedding(d);
    int v_i, v_j;
    float i_j_dot, w_ij;
    for(int iter_id = 0; iter_id < max_iter; iter_id++)
    {
        if(verbose)
        {
            if(iter_id%100 == 0)
            {
                std::cout << "\tIter id: " << iter_id << std::endl;
                _print_f_value(d);
            }
        }
        for(int edge_id = 0; edge_id < n_edges; edge_id++)
        {
            v_i = edges[edge_id][0];
            v_j = edges[edge_id][1];
            w_ij = edge_weights[edge_id];
            if(v_j <= v_i)
                continue;
            i_j_dot = 0;
            for(int d_id = 0; d_id < d; d_id++)
                i_j_dot += X[v_i][d_id]*X[v_j][d_id];
            for(int d_id = 0; d_id < d; d_id++)
                X[v_i][d_id] -= eta*(regu*X[v_i][d_id] - (w_ij - i_j_dot)*X[v_j][d_id]);
        }
    }
    t = clock() - t;
    t = clock();
    saveEmbToTxt(of_name, d);
}

BOOST_PYTHON_MODULE(graphFac_ext)
{
    // Py_Initialize();
    // Py_Initialize();
    // np::initialize();
    // srand(time(NULL));
    p::def("learn_embedding", learn_embedding);
}

