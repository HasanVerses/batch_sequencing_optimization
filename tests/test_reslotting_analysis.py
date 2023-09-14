import networkx as nx
import numpy as np

from opt.algorithms.reslotting import unoptimized, encode_destination_nodes, valid_initialization
from opt.graph import get_distance, add_shortest_paths, add_distances
from opt.tasks import random_graph_and_reslotting_task
from opt.api.reslotting import analyze_swaps



def test_unoptimized():

    _, _, distance = unoptimized(G_rand, rand_sources, rand_destinations, None, None, cumulative_distances=False)

    assert sum([
        get_distance(G_rand, rand_sources[idx-1], rand_destinations[idx-1]) \
            + get_distance(G_rand, rand_destinations[idx-1], rand_sources[idx]) 
            for idx in range(1,len(rand_destinations))
        ]) + get_distance(
                G_rand, 
                rand_sources[len(rand_destinations)-1], 
                rand_destinations[len(rand_destinations)-1]
            ) == distance


def test_analyze_swaps():

    G.bins = {chr(x): {"pos": (0, 0), "closest_waypoint": x-65, "distance": 0.1} for x in range(65,76)}

    test_sources = ["A", "B", "G", "E", "C", "D"]
    test_destinations = ["B", "A", "E", "G", "D", "C"]

    distance, sequence, path = analyze_swaps(G, [test_sources, test_destinations], cumulative_distances=False, output_df=False)
    
    double_traversed_edge_weights = [get_distance(G, 0, 1)*2, get_distance(G, 2, 3)*2, get_distance(G, 6, 4)*2]
    single_traversed_edge_weights = [get_distance(G, 0, 6), get_distance(G, 6, 2)]

    extra_distance = G.bins["A"]["distance"]*3 \
        + G.bins["B"]["distance"]*2 \
        + G.bins["G"]["distance"]*4 \
        + G.bins["E"]["distance"]*2 \
        + G.bins["C"]["distance"]*3 \
        + G.bins["D"]["distance"]*2

    assert np.isclose(sum(single_traversed_edge_weights) + sum(double_traversed_edge_weights) + extra_distance, distance)
    assert [0, 1, 0, 1, 2, 3, 4, 5, 6, 5, 4, 5, 6, 5, 4, 3, 2, 3, 2] == path
    assert [
        "Pick item from A", 
        "Place item from A in B", 
        "Pick item from B", 
        "Place item from B in A", 
        "Pick item from G", 
        "Place item from G in E", 
        "Pick item from E", 
        "Place item from E in G", 
        "Pick item from C", 
        "Place item from C in D", 
        "Pick item from D", 
        "Place item from D in C"] == sequence.english


G_rand, _, _, rand_sources, rand_destinations = random_graph_and_reslotting_task(num_items=4, task_type="swap")

rand_encoded_destinations = encode_destination_nodes(rand_sources, rand_destinations)
rand_initial_state = valid_initialization(rand_sources, rand_encoded_destinations)


num_nodes = 40

G = nx.Graph()
for n in range(num_nodes):
    G.add_node(n)
    if n > 0:
        G.add_edge(n,n-1, weight=np.random.rand())
    add_shortest_paths(G)
    add_distances(G)