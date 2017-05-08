// #include <boost/python.hpp>
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

#ifdef __cplusplus
extern "C"
{
#endif
   #include <cblas.h>
#ifdef __cplusplus
}
#endif

// using namespace boost:python;
using namespace std;

#define NUM_ITER 1000

int n_nodes;
int n_edges;
int **edges;
float eta = 0.1;
float regu = 0.01;
float **X;
int d = 100;

// BOOST_PYTHON_MODULE(learn_embedding_gf)
// {
//     // Add regular functions to the module.
//     def("learn_embedding", learn_embedding);
// }

void load_network()
{
    ifstream f;
    char ch;
    int v_i, v_j, row;

    f.open("kabutar.txt");
    f >> n_nodes;
    f >> n_edges;
    edges = (int **)malloc(n_edges*sizeof(int *));
    for(int i = 0; i < n_edges; i++)
        edges[i] = (int *)malloc(2*sizeof(int));

    X = (float **)malloc(n_nodes*sizeof(float *));
    for(int i = 0; i < n_nodes; i++)
        X[i] = (float *)malloc(d*sizeof(float));

    assert(edges != NULL);
    assert(X != NULL);

    row = 0;
    while(f >> v_i >> ch >> v_j)
    {
        edges[row][0] = v_i;
        edges[row][1] = v_j;
        row++;
    }
}

void gen_rand_network(int n, int m)
{
    int v_i, v_j, row;
    n_nodes = n;
    n_edges = m;
    edges = (int **)malloc(n_edges*sizeof(int *));
    for(int i = 0; i < n_edges; i++)
        edges[i] = (int *)malloc(2*sizeof(int));

    X = (float **)malloc(n_nodes*sizeof(float *));
    for(int i = 0; i < n_nodes; i++)
        X[i] = (float *)malloc(d*sizeof(float));

    assert(edges != NULL);
    assert(X != NULL);
    std::cout << "Assigned memory to edges and embedding" << std::endl;
    row = 0;
    while(row < n_edges)
    {
        edges[row][0] = rand()%n_nodes;
        edges[row][1] = rand()%n_nodes;
        row++;
    }
}

void init_embedding()
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator (seed);
    uniform_real_distribution<double> distribution (0.0,1.0);
    for (int i = 0; i < n_nodes; i++)
        for(int j = 0; j < d; j++)
            X[i][j] = distribution(generator);
}

void learn_embedding(std::string f_name, int d, float eta, float regu, float max_iter)
{
    int v_i, v_j;
    float i_j_dot;
    for(int iter_id = 0; iter_id < NUM_ITER; iter_id++)
    {
        if(iter_id%100 == 0)
            std::cout << "Iter id: " << iter_id << std::endl;
        for(int edge_id = 0; edge_id < n_edges; edge_id++)
        {
            v_i = edges[edge_id][0];
            v_j = edges[edge_id][1];
            if(v_j >= v_i)
                continue;
            i_j_dot = cblas_sdot(d, X[v_i], 1, X[v_j], 1);
            // cblas_scopy(d, X[v_i], 1, blas_temp, 1);
            cblas_scal(d, -eta*regu, X[v_i]);
            cblas_saxpy(d, eta*(w_ij - i_j_dot), X[v_j], 1, X[v_i], 1)
            // i_j_dot = 0;
            // for(int d_id = 0; d_id < d; d_id++)
            //     i_j_dot += X[v_i][d_id]*X[v_j][d_id];
            // for(int d_id = 0; d_id < d; d_id++)
            //     X[v_i][d_id] -= regu*X[v_i][d_id] - (1 - i_j_dot)*X[v_j][d_id];
        }
    }
    return X;
}

int main(int argc, char *argv[])
{
    clock_t t;
    srand(time(NULL));

    std::cout << "Generating network" << std::endl;
    gen_rand_network(2e6, 1e7);

    std::cout << "Randomly initialize embedding matrix" << std::endl;
    init_embedding();

    t = clock();
    std::cout << "Learning embedding" << std::endl;
    learn_embedding();
    t = clock() - t;
    std::cout << "Learnt embedding. Time taken:" << ((float)t)/CLOCKS_PER_SEC << " secs" << std::endl;
}