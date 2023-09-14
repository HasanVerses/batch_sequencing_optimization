import networkx as nx

from opt.graph import random_graph, randomly_locally_connect, spatial_neighbors



def test_spatial_neighbors():
    middle_node_neighbors = spatial_neighbors(G, 10)
    assert [9, 11, 18, 2] == middle_node_neighbors
    corner_node_neighbors = spatial_neighbors(G, 0)
    assert [1, 8] == corner_node_neighbors


def test_randomly_locally_connect():
    G_copy = nx.create_empty_copy(G)
    G_copy.dimensions = G.dimensions
    randomly_locally_connect(G_copy)
    assert nx.is_connected(G_copy)


def test_random_graph():
    G = random_graph()
    assert (5, 8) == G.dimensions
    assert nx.is_connected(G)
    assert 40 == len(list(G)) 


G = random_graph()
