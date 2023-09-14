import numpy as np
import networkx as nx

from opt.graph import get_distance_multi, add_shortest_paths, add_distances
from opt.algorithms.common import energy, valid_initial_state, parse_constraint



def test_energy():
    assert get_distance_multi(G, [start] + locations + [end]) == energy(G, locations, start, end)


def test_energy_with_constraints():
    num_priority_1_nodes = 2
    num_priority_2_nodes = 5
    priority_1_nodes = set(np.random.choice(locations, num_priority_1_nodes, replace=False))
    not_in_priority_1 = [x for x in locations if x not in priority_1_nodes and x != start and x != end]
    priority_2_nodes = set(np.random.choice(not_in_priority_1, num_priority_2_nodes, replace=False))
    constraint = [priority_1_nodes, priority_2_nodes]
    compiled_constraint_fn, fn_kwargs = parse_constraint(constraint)

    good_state = valid_initial_state(locations, constraint)
    bad_state = good_state[1:] + [good_state[0]]

    good_energy = energy(G, good_state, start, end, constraint_fn=compiled_constraint_fn, constraint_penalty=100, constraint_fn_kwargs=fn_kwargs)
    bad_energy = energy(G, bad_state, start, end, constraint_fn=compiled_constraint_fn, constraint_penalty=100, constraint_fn_kwargs=fn_kwargs)

    assert good_energy < bad_energy


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
