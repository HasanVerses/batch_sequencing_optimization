import numpy as np
import networkx as nx

from opt.graph import add_shortest_paths, add_distances, get_closest_waypoint_multi
from opt.algorithms.naive import naive_tsp
from opt.tasks import random_graph_and_reslotting_task
from opt.algorithms.reslotting import (
    valid_initialization, 
    encode_destination_nodes, 
    decode_destination_nodes, 
    reslotting_constraints,
)
from opt.api.utils.reslotting import bin_role_map



def test_bin_role_map():
    sources, destinations = ["A", "C", "B", "D"], ["C", "A", "D", "B"]
    source_waypoints = get_closest_waypoint_multi(G, sources)
    destination_waypoints = get_closest_waypoint_multi(G, destinations)

    encoded_destination_nodes = encode_destination_nodes(source_waypoints, destination_waypoints, suffix="_d.", prefix="_o.")

    sequence = valid_initialization(source_waypoints, encoded_destination_nodes)

    assert ["A", "_o.C_d.A", "C", "_o.A_d.C", "B", "_o.D_d.B", "D", "_o.B_d.D"] == bin_role_map(
        sequence, 
        sources, 
        destinations,
        source_waypoints,
        encoded_destination_nodes, 
        encode_destination_nodes
    )


def test_reslot_sequencing_naive_with_minimal_cart_capacity():

    seq, _, _ = naive_tsp(
        G_rand, 
        locations=rand_sources + rand_encoded_destinations, 
        decoder=decode_destination_nodes,
        constraint_fn=reslotting_constraints
    )

    assert all(reslotting_constraints(seq))


def test_reslot_sequencing_naive_with_larger_cart_capacity():

    constraint_fn_kwargs = {"num_slots": 8}

    seq, _, _ = naive_tsp(
        G_rand, 
        locations=rand_sources + rand_encoded_destinations, 
        decoder=decode_destination_nodes,
        constraint_fn=reslotting_constraints,
        constraint_fn_kwargs=constraint_fn_kwargs
    )

    assert all(reslotting_constraints(seq, **constraint_fn_kwargs))


def test_reslot_sequencing_naive_distance():

    seq, _, _ = naive_tsp(
        G, 
        locations=sources + encoded_destinations, 
        start=start,
        decoder=decode_destination_nodes,
        constraint_fn=reslotting_constraints,
    )
    assert [0, 1, 13, 11, 4, 8, 2] == decode_destination_nodes(seq)

    constraint_fn_kwargs = {"num_slots": 2}

    seq, _, _ = naive_tsp(
        G, 
        locations=sources + encoded_destinations, 
        start=start,
        decoder=decode_destination_nodes,
        constraint_fn=reslotting_constraints,
        constraint_fn_kwargs=constraint_fn_kwargs
    )
    decoded_sequence = decode_destination_nodes(seq)
    assert decoded_sequence.index(8) < decoded_sequence.index(4) 


G_rand, _, _, rand_sources, rand_destinations = random_graph_and_reslotting_task(num_items=4, task_type="swap")

rand_encoded_destinations = encode_destination_nodes(rand_sources, rand_destinations)
rand_initial_state = valid_initialization(rand_sources, rand_encoded_destinations)


num_nodes = 14

G = nx.Graph()
for n in range(num_nodes):
    G.add_node(n)
    if n > 0:
        G.add_edge(n,n-1, weight=np.random.rand())
    add_shortest_paths(G)
    add_distances(G)

G.bins = {chr(x): {"pos": (0, 0), "closest_waypoint": x-65, "distance": 0.1} for x in range(65,76)}


sources = [8, 11, 1]
destinations = [2, 4, 13]
start = 0

encoded_destinations = encode_destination_nodes(sources, destinations)
initial_state = valid_initialization(sources, encoded_destinations)
