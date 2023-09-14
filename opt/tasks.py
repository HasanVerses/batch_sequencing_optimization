import numpy as np
from opt.graph import random_graph
from opt.algorithms.reslotting import encode_destination_nodes



def random_task(graph, num_nodes, return_indices=False):
    """Return a random start node, end node, and intermediate nodes to visit (or their indices), given a graph."""
    node_list = list(np.random.choice(graph.number_of_nodes(), num_nodes, replace=False))
    if not return_indices:
        actual_nodes = list(graph)
        node_list = [actual_nodes[idx] for idx in node_list]
        
    return node_list[0], node_list[-1], node_list[1:-1]


def random_graph_and_task(num_nodes=45, return_indices=False):
    """Return a random graph and start, end and intermediate nodes to visit on it."""
    graph = random_graph(max(60, num_nodes*4))
    return graph, *random_task(graph, num_nodes, return_indices)


def random_reslotting_task(graph, num_items, task_type="generic", map_to_virtual_destinations=False, return_indices=False, suffix='_d.', prefix='_o.'):
    """
    Return a set of source and destination nodes representing a task, given a 
    number of items, a graph, and a task type.

    `task_type`: "generic" - Sample sources and destinations independently; may be overlap
                 "shuffle" - Destinations are a permutation of sources (with source[i] != destination[i])
                 "swap"    - Each pair of items are source and destination for each other, as in the original task

    Returns (start_node, end_node, source_nodes, destination_nodes)
    """

    assert task_type in ["generic", "shuffle", "swap"], "Available task types are `generic` and `swap`"
    
    num_items_range = range(num_items)
    #visible_graph = visible_nodes(graph)
    actual_nodes = list(graph)    
    
    start = np.random.choice(graph.number_of_nodes())
    end = np.random.choice(graph.number_of_nodes())
    if not return_indices:
        start = actual_nodes[start]
        end = actual_nodes[end]

    if task_type == "generic":
        sources = destinations = list(np.random.choice(graph.number_of_nodes(), num_items, replace=False))
        while any([sources[idx] == destinations[idx] for idx in num_items_range]):
            destinations = list(np.random.choice(graph.number_of_nodes(), num_items, replace=False))
        
    elif task_type == "shuffle":
        sources = destinations = list(np.random.choice(graph.number_of_nodes(), num_items, replace=False))
        while any([sources[idx] == destinations[idx] for idx in num_items_range]):
            destinations = np.random.permutation(sources)
                        
    elif task_type == "swap":
        assert num_items % 2 == 0, "Swap task requires an even number of items"
        nodes = list(np.random.choice(graph.number_of_nodes(), num_items, replace=False))
        sources = []
        destinations = []
        while nodes:
            source_idx, destination_idx = np.random.choice(len(nodes), 2, replace=False)
            source = nodes[source_idx]
            destination = nodes[destination_idx]
            nodes = [node for idx, node in enumerate(nodes) if idx not in [source_idx, destination_idx]]
            sources += [source, destination]
            destinations += [destination, source]
    
    if not return_indices:
        sources = [actual_nodes[idx] for idx in sources]
        destinations = [actual_nodes[idx] for idx in destinations]

    if map_to_virtual_destinations:
        destinations = encode_destination_nodes(sources, destinations, suffix, prefix)
    return start, end, sources, destinations


def random_graph_and_reslotting_task(
    num_nodes=40, 
    num_items=8, 
    task_type="generic", 
    map_to_virtual_destinations=False, 
    return_indices=True,
    suffix='_d.',
    prefix='_o.'
):
    """Return a random graph and start, end and intermediate nodes to visit on it."""
    graph = random_graph(num_nodes)
    return graph, *random_reslotting_task(graph, num_items, task_type, map_to_virtual_destinations, return_indices, suffix, prefix)
