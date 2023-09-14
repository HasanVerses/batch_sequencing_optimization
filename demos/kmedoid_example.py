import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse

from opt.graph import random_graph
from opt.algorithms.kmedoids import run_k_medoids_graph


def run(k_max = 5, num_iter = 10, N = 30, path_algo = 'default', verbose = False):

    """
    Simple run function that creates a random locally-connected spatial graph, computes a shortest-path-length matrix
    for all pairs of nodes, and then passes that + other parameters into the k_medoids partitioning algorithm.
    @TODO: Should add some extra lines at the end that generate outputs, similar to how Alex does it for his demos (either animations or images or something).
    
    """

    G = random_graph(30)

    A = nx.to_numpy_array(G)
    if path_algo == 'default':
        all_spls = dict(nx.all_pairs_shortest_path_length(G))    # all_spls[i][j] contains shortest path length between node i and node j
        D = np.zeros_like(A)
        # populate shortest paths martix using outputs of nx.all_pairs_shortest_path_length
        for i, node1 in enumerate(G.nodes()):
            for j, node2 in enumerate(G.nodes()):
                try:
                    D[i,j] = all_spls[node1][node2]
                except:
                    D[i,j] = np.inf
    elif path_algo == 'f_w':
        D = nx.floyd_warshall_numpy(G) # this uses something called Floyd-Warshall algorithm, somehow different than Djikstra's. Convenient option because it gives a matrix (N x N) output

    # run the k-medoids algorithm
    medoids, id_vector = run_k_medoids_graph(D, k_max, num_iter, verbose = verbose)

    if verbose:
        print(f"Final number of identified clusters: {len(medoids)}\n")

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clusters', '-nc', type = int,
                help = "Number of clusters",
                dest = "k_max", default=5)
    parser.add_argument('--num_iter', '-ni', type = int,
                help = "Number of iterations to run k-medoids for",
                dest = "num_iter", default=10)
    parser.add_argument('--N', '-N', type = str,
                help = "Number of nodes in the random graph to create",
                dest = "N", default=30)
    parser.add_argument('--path_algo', '-pa', type = str,
                help = "Shortest path algorithm",
                dest = "path_algo", default='default') 
    parser.add_argument('--verbose', action = argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=False)
    
    args = parser.parse_args()

    run(**vars(args))


