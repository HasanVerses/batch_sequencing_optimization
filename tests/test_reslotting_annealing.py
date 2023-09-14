import numpy as np
import networkx as nx

from opt.graph import add_shortest_paths, add_distances
from opt.algorithms.annealing import anneal, deterministic_anneal, anneal_thermodynamic
from opt.tasks import random_graph_and_reslotting_task
from opt.algorithms.reslotting import (
    valid_initialization, 
    encode_destination_nodes, 
    decode_destination_nodes, 
    reslotting_constraints
)
from opt.api.reslotting import optimize_swaps



def test_valid_initialization():
    assert all(reslotting_constraints(initial_state))


def test_encode_and_decode():
    assert destinations == decode_destination_nodes(encode_destination_nodes(sources, destinations))


def test_reslot_sequencing_with_annealing_minimal_cart_capacity():

    seq, _, _ = anneal(
        G, 
        locations=sources + encoded_destinations, 
        initial_state=initial_state, 
        decoder=decode_destination_nodes,
        constraint_fn=reslotting_constraints
    )

    assert all(reslotting_constraints(seq))


def test_reslot_sequencing_with_annealing_minimal_cart_capacity_deterministic():

    seq, _, _ = deterministic_anneal(
        G, 
        locations=sources + encoded_destinations, 
        initial_state=initial_state, 
        decoder=decode_destination_nodes,
        constraint_fn=reslotting_constraints
    )

    assert all(reslotting_constraints(seq))


def test_reslot_sequencing_with_annealing_larger_cart_capacity():

    seq, _, _ = anneal(
        G, 
        locations=sources + encoded_destinations, 
        initial_state=initial_state, 
        decoder=decode_destination_nodes,
        constraint_fn=reslotting_constraints,
        constraint_fn_kwargs=constraint_fn_kwargs
    )

    assert all(reslotting_constraints(seq, **constraint_fn_kwargs))


def test_reslot_sequencing_with_annealing_larger_cart_capacity_deterministic():

    seq, _, _ = deterministic_anneal(
        G, 
        locations=sources + encoded_destinations, 
        initial_state=initial_state, 
        decoder=decode_destination_nodes,
        constraint_fn=reslotting_constraints,
        constraint_fn_kwargs=constraint_fn_kwargs
    )

    assert all(reslotting_constraints(seq, **constraint_fn_kwargs))


def test_reslot_sequencing_with_annealing_distance():

    seq, _, _ = anneal(
        G2, 
        locations=sources2 + encoded_destinations2, 
        initial_state=initial_state2, 
        start=start2,
        decoder=decode_destination_nodes,
        constraint_fn=reslotting_constraints,
    )
    assert [0, 1, 13, 11, 4, 8, 2] == decode_destination_nodes(seq)

    constraint_fn_kwargs = {"num_slots": 2}

    seq, _, _ = anneal(
        G2, 
        locations=sources2 + encoded_destinations2, 
        initial_state=initial_state2, 
        start=start2,
        decoder=decode_destination_nodes,
        constraint_fn=reslotting_constraints,
        constraint_fn_kwargs=constraint_fn_kwargs
    )
    decoded_sequence = decode_destination_nodes(seq)
    assert decoded_sequence.index(8) < decoded_sequence.index(4) 


def test_reslotting_annealing_with_api_minimal_capacity():
    
    G2.bins = {chr(x): {"pos": (0, 0), "closest_waypoint": x-65, "distance": 0.1} for x in range(65,76)}

    test_sources = ["A", "B", "G", "E", "C", "D"]
    test_destinations = ["B", "A", "E", "G", "D", "C"]

    _, sequence, _ = optimize_swaps(G2, [test_sources, test_destinations], cart_capacity=1, constraint_weight=100, constraint_weight_base=None, output_csv_format=None)

    assert all(reslotting_constraints(sequence.raw, **constraint_fn_kwargs))


def test_reslotting_annealing_with_api_larger_capacity():
    
    G2.bins = {chr(x): {"pos": (0, 0), "closest_waypoint": x-65, "distance": 0.1} for x in range(65,76)}

    test_sources = ["A", "B", "G", "E", "C", "D"]
    test_destinations = ["B", "A", "E", "G", "D", "C"]

    _, sequence, _ = optimize_swaps(G2, [test_sources, test_destinations], constraint_weight=100, constraint_weight_base=None, output_csv_format=None)

    assert all(constraint_fn(sequence.raw, **{"num_slots": 8}))


def test_reslot_sequencing_with_annealing_thermodynamic_larger_cart_capacity():

    seq, _, _ = anneal_thermodynamic(
        G, 
        locations=sources + encoded_destinations, 
        initial_state=initial_state, 
        decoder=decode_destination_nodes,
        constraint_fn=reslotting_constraints,
        constraint_fn_kwargs=constraint_fn_kwargs
    )

    assert all(reslotting_constraints(seq, **constraint_fn_kwargs))


G, start, end, sources, destinations = random_graph_and_reslotting_task(task_type="swap")

encoded_destinations = encode_destination_nodes(sources, destinations)
initial_state = valid_initialization(sources, encoded_destinations)


num_nodes = 14

G2 = nx.Graph()
for n in range(num_nodes):
    G2.add_node(n)
    if n > 0:
        G2.add_edge(n,n-1, weight=np.random.rand())
    add_shortest_paths(G2)
    add_distances(G2)

sources2 = [8, 11, 1]
destinations2 = [2, 4, 13]
start2 = 0

encoded_destinations2 = encode_destination_nodes(sources2, destinations2)
initial_state2 = valid_initialization(sources2, encoded_destinations2)

constraint_fn = reslotting_constraints
constraint_fn_kwargs = {"num_slots": 1}
