import networkx as nx

from opt.tasks import random_graph_and_task
from opt.graph import (
    get_shortest_path,
    get_shortest_path_multi, 
    get_distance, 
    get_distance_multi,
    get_closest_waypoint,
    get_closest_waypoint_multi
)



def test_get_shortest_path():
    expected = nx.dijkstra_path(G, start, end, weight='weight')
    actual = get_shortest_path(G, start, end)
    assert expected == actual


def test_get_distance():
    expected = nx.shortest_path_length(G, start, end, weight='weight')
    actual = get_distance(G, start, end)
    assert expected == actual


def test_get_distance_multi():
    expected = sum(
        [nx.shortest_path_length(
            G, locations[idx], 
            locations[idx+1], 
            weight='weight'
        ) for idx in range(len(locations) - 1)])
    actual = get_distance_multi(G, locations)
    assert expected == actual


def test_get_shortest_path_multi():
    expected = []
    for idx in range(len(locations) - 1):
        expected += nx.dijkstra_path(G, locations[idx], locations[idx+1], weight='weight')[:-1]
    expected += [locations[-1]]
    actual = get_shortest_path_multi(G, locations)
    assert expected == actual


def test_get_closest_waypoint():
    assert 2 == get_closest_waypoint(G, "C")


def test_get_closest_waypoint_multi():
    assert [0, 2, 4] == get_closest_waypoint_multi(G, ["A", "C", "E"])


num_locations=30
G, start, end, locations = random_graph_and_task(num_nodes=num_locations)
G.bins = {chr(x): {"pos": (0, 0), "closest_waypoint": x-65, "distance": 0.1} for x in range(65,76)}
