from opt.defaults.cost import GraphDistanceCost
from opt.tasks import random_graph_and_task
from opt.graph import get_distance_multi, get_node_list



def test_graph_distance_class():
    distance_cost = GraphDistanceCost(G)
    assert get_distance_multi(G, nodes) == distance_cost.eval(nodes)


G, start, end, locations = random_graph_and_task(30)
nodes = get_node_list(locations, start, end)
