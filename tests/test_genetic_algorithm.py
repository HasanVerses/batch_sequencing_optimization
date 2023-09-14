import networkx as nx
import numpy as np

from opt.api.core import solve_tsp
from opt.api.snaking import optimize_picks
from opt.graph import add_shortest_paths, add_distances
from opt.algorithms.genetic import genetic, genetic_crossover
from opt.algorithms.common import parse_constraint



def test_genetic():
    sequence, path, distance = genetic(G, locations, start=start, end=end)
    assert [start] + sorted(locations) + [end] == sequence
    assert list(G) == path
    edge_weights = list(nx.get_edge_attributes(G, 'weight').values())
    assert np.isclose(
        sum(edge_weights), 
        distance
    )


def test_api_with_genetic():
    sequence, path, distance = solve_tsp(G, locations, start=start, end=end, algorithm="genetic")
    assert [start] + sorted(locations) + [end] == sequence
    assert list(G) == path
    edge_weights = list(nx.get_edge_attributes(G, 'weight').values())
    assert np.isclose(
        sum(edge_weights), 
        distance
    )


def test_genetic_crossover():
    sequence, path, distance = genetic_crossover(G, locations, start=start, end=end)
    assert [start] + sorted(locations) + [end] == sequence
    assert list(G) == path
    edge_weights = list(nx.get_edge_attributes(G, 'weight').values())
    assert np.isclose(
        sum(edge_weights), 
        distance
    )


def test_api_with_genetic_crossover():
    sequence, path, distance = solve_tsp(G, locations, start=start, end=end, algorithm="genetic x")
    assert [start] + sorted(locations) + [end] == sequence
    assert list(G) == path
    edge_weights = list(nx.get_edge_attributes(G, 'weight').values())
    assert np.isclose(
        sum(edge_weights), 
        distance
    )

def test_api_with_genetic_constrained():
    priority_1_nodes = set(np.random.choice(locations, 2, replace=False))
    not_in_priority_1 = [x for x in locations if x not in priority_1_nodes and x != start and x != end]
    priority_2_nodes = set(np.random.choice(not_in_priority_1, 5, replace=False))
    constraint = [priority_1_nodes, priority_2_nodes]

    sequence, _, _ = solve_tsp(
        G, 
        locations, 
        start=start, 
        end=end, 
        algorithm="genetic", 
        constraint=constraint
    )
    compiled_constraint_fn, fn_kwargs = parse_constraint(constraint)
    assert compiled_constraint_fn(sequence[1:-1], **fn_kwargs)


def test_api_with_genetic_constraint_crossover():
    priority_1_nodes = set(np.random.choice(locations, 2, replace=False))
    not_in_priority_1 = [x for x in locations if x not in priority_1_nodes and x != start and x != end]
    priority_2_nodes = set(np.random.choice(not_in_priority_1, 5, replace=False))
    constraint = [priority_1_nodes, priority_2_nodes]

    sequence, _, _ = solve_tsp(
        G, 
        locations, 
        start=start, 
        end=end, 
        algorithm="genetic x", 
        constraint=constraint
    )
    compiled_constraint_fn, fn_kwargs = parse_constraint(constraint)
    assert compiled_constraint_fn(sequence[1:-1], **fn_kwargs)


def test_genetic_with_bin_data():

    G.bins = {chr(x): {"pos": (0, 0), "closest_waypoint": x-65, "distance": 0.1} for x in range(65,76)}

    test_sequence = ["A", "E", "B", "D", "G"]

    distance, sequence, path = optimize_picks(G, test_sequence, fixed_start=True, fixed_end=True, algorithm="genetic", output_df=False, cumulative_distances=False)
    relevant_edge_weights = list(nx.get_edge_attributes(G, 'weight').values())[:6]

    extra_distance = G.bins["A"]["distance"] \
        + G.bins["B"]["distance"]*2 \
        + G.bins["D"]["distance"]*2 \
        + G.bins["E"]["distance"]*2 \
        + G.bins["G"]["distance"]

    assert np.isclose(sum(relevant_edge_weights) + extra_distance, distance)
    assert list(range(7)) == path
    assert ["A", "B", "D", "E", "G"] == sequence


def test_genetic_crossover_with_bin_data():

    G.bins = {chr(x): {"pos": (0, 0), "closest_waypoint": x-65, "distance": 0.1} for x in range(65,76)}

    test_sequence = ["A", "E", "B", "D", "G"]

    distance, sequence, path = optimize_picks(G, test_sequence, fixed_start=True, fixed_end=True, algorithm="genetic x", output_df=False, cumulative_distances=False)
    relevant_edge_weights = list(nx.get_edge_attributes(G, 'weight').values())[:6]

    extra_distance = G.bins["A"]["distance"] \
        + G.bins["B"]["distance"]*2 \
        + G.bins["D"]["distance"]*2 \
        + G.bins["E"]["distance"]*2 \
        + G.bins["G"]["distance"]

    assert np.isclose(sum(relevant_edge_weights) + extra_distance, distance)
    assert list(range(7)) == path
    assert ["A", "B", "D", "E", "G"] == sequence


num_nodes = 40

G = nx.Graph()
for n in range(num_nodes):
    G.add_node(n)
    if n > 0:
        G.add_edge(n,n-1, weight=np.random.rand())
    add_shortest_paths(G)
    add_distances(G)

num_locations = 8

locations = list(np.random.choice(num_nodes - 2, num_locations, replace=False) + 1)

start = 0
end = num_nodes - 1