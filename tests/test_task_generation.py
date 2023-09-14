from opt.graph import random_graph
from opt.tasks import random_task, random_graph_and_task, random_reslotting_task, random_graph_and_reslotting_task



def test_random_task():

    start, end, locations = random_task(G, num_locations)

    assert start in G
    assert end in G
    assert all([location in G for location in locations])
    assert num_locations == len([start] + locations + [end])


def test_random_graph_and_task():
    graph, start, end, locations = random_graph_and_task(num_nodes=num_locations)

    assert start in graph
    assert end in graph
    assert all([location in graph for location in locations])
    assert num_locations == len([start] + locations + [end])
    assert max(60, num_locations*4) == len(list(graph))


def test_random_reslotting_task():

    start, end, sources, destinations = random_reslotting_task(G, num_locations, task_type="shuffle")

    assert start in G
    assert end in G
    assert all([source in G for source in sources])
    assert all([destination in G for destination in destinations])
    assert num_locations*2 + 2 == len([start] + sources + destinations + [end])
    assert all([source in destinations for source in sources])

    start, end, sources, destinations = random_reslotting_task(G, num_locations, task_type="swap")

    assert sources[0] == destinations[1]
    assert destinations[0] == sources[1]


def test_random_graph_and_reslotting_task():
    graph, start, end, sources, destinations = random_graph_and_reslotting_task(num_items=num_locations)

    assert start in graph
    assert end in graph
    assert all([source in graph for source in sources])
    assert all([destination in graph for destination in destinations])
    assert num_locations*2 == len(sources + destinations)


G = random_graph()
num_locations = 30
