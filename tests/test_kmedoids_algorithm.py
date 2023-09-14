import numpy as np
import networkx as nx

from opt.graph import random_graph, add_shortest_paths, add_distances
from opt.algorithms.kmedoids import run_k_medoids_graph


def test_kmedoids_one_cluster():
    """ 
    Test to make sure that if you specify one cluster and the graph
    is fully connected (obvious case), the kmedoids algorithm runs appropriately and finds one cluster
    """

    n = 5

    G = nx.complete_graph(n)

    all_spls = dict(nx.all_pairs_shortest_path_length(G))    # all_spls[i][j] contains shortest path length between node i and node j
    D = np.zeros((n, n))
    # populate shortest paths martix using outputs of nx.all_pairs_shortest_path_length
    for i, node1 in enumerate(G.nodes()):
        for j, node2 in enumerate(G.nodes()):
            try:
                D[i,j] = all_spls[node1][node2]
            except:
                D[i,j] = np.inf
    
    k, num_iter = 1, 10
    # run the k-medoids algorithm
    medoids, id_vector = run_k_medoids_graph(D, k, num_iter, verbose = False)

    assert len(medoids) == 1

def test_kmedoids_two_clusters_SBM():
    """ 
    Test to make sure that if you specify 2 clusters
    and the system is well described with two communities (by construction via a stochastic block model),
    then the kmedoids algorithm finds the two clusters you would expect.
    """

    sizes = [10, 10]

    p = [[0.95, 0.05], [0.05, 0.95]]

    G = nx.stochastic_block_model(sizes, p)

    n = G.number_of_nodes()

    all_spls = dict(nx.all_pairs_shortest_path_length(G))    # all_spls[i][j] contains shortest path length between node i and node j
    D = np.zeros((n, n))
    # populate shortest paths martix using outputs of nx.all_pairs_shortest_path_length
    for i, node1 in enumerate(G.nodes()):
        for j, node2 in enumerate(G.nodes()):
            try:
                D[i,j] = all_spls[node1][node2]
            except:
                D[i,j] = np.inf
    
    k, num_iter = 2, 10
    # run the k-medoids algorithm
    medoids, id_vector = run_k_medoids_graph(D, k, num_iter, verbose = False)

    assert len(medoids) == 2
