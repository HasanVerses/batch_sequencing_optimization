import math
import random
import networkx as nx
import numpy as np
import pandas as pd
import time



def find_num_columns_for_layout(num_nodes):
    """Helper function to figure out random graph layout"""
    cols = int(math.sqrt(num_nodes))
    while not num_nodes//cols == num_nodes/cols:
        cols += 1
    
    return cols


def define_undirected_graph(num_nodes=40):
    """Generate an undirected graph that is as close to square as possible given an even number of nodes"""
    assert not num_nodes%2, "Please use an even number of nodes, at least"
    num_cols = find_num_columns_for_layout(num_nodes)
    
    G = nx.Graph()    
    for n in range(num_nodes):
        G.add_node(n, pos = (n%num_cols, n//num_cols))
    
    nx.set_node_attributes(G, "#AAB4D9", name="node_color")
    
    G.dimensions = (num_nodes//num_cols, num_cols)
        
    return G


def add_shortest_paths(graph):
    """Add shortest paths between all points (computed using Dijkstra's algorithm) as a property to the input graph."""
    graph.shortest_paths = {k: v for (k, v) in list(nx.all_pairs_dijkstra_path(graph))} 


def add_distances(graph):
    """Add shortest distances (computed using Dijkstra's algorithm) as a property to the input graph."""
    graph.distances = {k: v for (k, v) in list(nx.all_pairs_dijkstra_path_length(graph))}


def is_left(graph, node):
    return node%graph.dimensions[1] == 0

def is_right(graph, node):
    return (node+1)%graph.dimensions[1] == 0

def is_top(graph, node):
    return node//graph.dimensions[1] == 0
    
def is_bottom(graph, node):
    return node//graph.dimensions[1] >= graph.dimensions[0] - 1
    

def spatial_neighbors(graph, node):
    neighbors = []
    if not is_left(graph, node):
        neighbors.append(node-1)
    if not is_right(graph, node):
        neighbors.append(node+1)
    if not is_bottom(graph, node):
        neighbors.append(node+graph.dimensions[1])
    if not is_top(graph, node):
        neighbors.append(node-graph.dimensions[1])
        
    return neighbors


def add_edge(graph, start, prob=0.5):
    if start%graph.dimensions[1] > 0 and np.random.choice(2, p=[1-prob, prob]):
        graph.add_edge(start,start-1)
    if start//graph.dimensions[1] > 0 and np.random.choice(2, p=[1-prob, prob]):
        graph.add_edge(start,start-graph.dimensions[1])

        
def randomly_locally_connect(graph, prob=0.5, fully_connect: bool=True):
    """
    Add local connections (i.e. to nodes above, below, to the left or right of a given node) with probability `prob`.
    If `fully_connect` is true, ensure that every node is reachable from every other node.
    
    returns: Nothing; modifies graph in place
    """


    for n in range(graph.number_of_nodes()):
        if n%graph.dimensions[1] > 0 and np.random.choice(2, p=[1-prob, prob]):
            graph.add_edge(n,n-1)
        if n//graph.dimensions[1] > 0 and np.random.choice(2, p=[1-prob, prob]):
            graph.add_edge(n,n-graph.dimensions[1])
    
    if fully_connect:
        while not nx.is_connected(graph):
            components = list(nx.connected_components(graph))
            components.sort(key=len)
            n = random.sample(list(components[0]),1)[0]
            neighbors = spatial_neighbors(graph, n) 
            graph.add_edge(n, random.sample(neighbors,1)[0])

            
def random_graph(num_nodes=40, connect_prob=0.75):
    """
    Return a connected rectangular spatial graph with `num_nodes` nodes and random local connections
    added with probability `connect_prob`. Add shortest paths and their distances."""
    G = define_undirected_graph(num_nodes)
    randomly_locally_connect(G, prob=connect_prob)
    add_shortest_paths(G)
    add_distances(G)
    
    return G


def get_distance_matrix(graph):
    nodes = list(graph)

    start_time = time.time()
    distances_df = pd.DataFrame(columns=nodes, index=nodes)
    distances = graph.distances if hasattr(graph, 'distances') else {k: v for (k, v) in list(nx.all_pairs_dijkstra_path_length(graph))}
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time to distances: {elapsed_time} seconds")

    # start_time = time.time()
    # for k in distances.keys():
    #     for v in distances[k].keys():
    #         distances_df.loc[k][v] = distances[k][v]
    #
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time to distances_df: {elapsed_time} seconds")

    distances_df = pd.DataFrame.from_dict(distances, orient='index')

    return distances_df


def get_shortest_path(graph, start, end, use_bins=False, use_cached_paths=True):
    if use_bins:
        assert hasattr(graph, "bins"), "Graph has no `bins` attribute"
        start = graph.bins[start]["closest_waypoint"]
        end = graph.bins[end]["closest_waypoint"]

    if use_cached_paths:
        assert hasattr(graph, "shortest_paths"), "Graph has no `shortest_paths` attribute"
        path = graph.shortest_paths[start].get(end)
        assert path, f"No path available to node {end}!"
    else:
        path = nx.shortest_path(graph, start, end, 'weight')

    return path


def get_distance(graph, start, end, use_bins=False):
    if use_bins:
        assert hasattr(graph, "bins"), "Graph has no `bins` attribute"

        start_bin_info = graph.bins[start]
        end_bin_info = graph.bins[end]

        start = start_bin_info["closest_waypoint"]
        end = end_bin_info["closest_waypoint"]
        if start == end:
            return 0

        extra_distance = start_bin_info["distance"] + end_bin_info["distance"]
    else:
        extra_distance = 0

    assert hasattr(graph, "shortest_paths"), "Graph has no `distances` attribute"
    distance = graph.distances[start].get(end)
    assert distance is not None, f"No distance record available for path from {start} to {end}!"

    return distance + extra_distance


def get_shortest_path_multi(graph, node_list, use_bins=False, use_cached_paths=True, return_destination_indices=False):
    if use_bins:
        assert hasattr(graph, "bins"), "Graph has no `bins` attribute"
        end_node = graph.bins[node_list[-1]]["closest_waypoint"]
    else:
        end_node = node_list[-1]
    
    if return_destination_indices:
        destination_indices = [0]

    path = []
    for idx in range(len(node_list)-1):
        next_segment = get_shortest_path(graph, node_list[idx], node_list[idx+1], use_bins, use_cached_paths)[:-1]
        path += next_segment
        if return_destination_indices:
            destination_indices.append(destination_indices[-1] + len(next_segment))

    path += [end_node]

    if return_destination_indices:
        return path, destination_indices
    else:
        return path


def get_distance_multi(graph, node_list, use_bins=False, cumulative_distance=False):
    distance = 0
    if cumulative_distance: cumulative_distances = [0]
    for idx in range(len(node_list)-1):
        this_distance = get_distance(graph, node_list[idx], node_list[idx+1], use_bins)
        distance += this_distance
        if cumulative_distance:
            cumulative_distances.append(distance)

    return distance if not cumulative_distance else cumulative_distances


def get_node_list(locations, start=None, end=None):
    """Return a list of nodes where start and end points may or may not be fixed"""
    locations = [location for location in locations if location not in [start, end]]
    start = [start] if start is not None else []
    end = [end] if end is not None else [] 
    return start + locations + end


def sort_by_iterable_len(v, target_list):
  """Utility for sorting connected components of a graph by their size"""
  lens = [len(x) for x in target_list]
  return lens[target_list.index(v)]


def prune_graph(graph):
    """Return a graph with only the largest connected component retained"""
    if not nx.is_connected(graph):
        print("Graph not connected. Pruning disconnected node(s)...")
        connected_components_list = list(nx.connected_components(graph))
        to_remove = sorted(
            connected_components_list, 
            key=lambda x: sort_by_iterable_len(x, connected_components_list)
        )[:-1]

        set_to_remove = set().union(*to_remove)
        graph.remove_nodes_from(set_to_remove)
    
    return graph


def waypoint_to_index(graph, space):
    assert hasattr(graph, "original_nodes"), "Graph has no node-index mapping!"
    return graph.original_nodes.index(space)


def index_to_waypoint(graph, idx):
    assert hasattr(graph, "original_nodes"), "Graph has no node-index mapping!"
    return graph.original_nodes[idx]


def get_closest_waypoint(graph, bin_name):
    assert hasattr(graph, "bins"), "Graph has no `bins` attribute"
    record = graph.bins.get(bin_name)
    assert record is not None, f"No bin record for {bin_name}!"

    waypoint = record.get("closest_waypoint")
    if waypoint is None:
        print(f"WARNING: No closest waypoint found for bin {bin_name}.")

    return waypoint


def get_closest_waypoint_multi(graph, bin_names, quit_on_missing_waypoint=True):
    waypoints = []
    for bin_name in bin_names:
        waypoint = get_closest_waypoint(graph, bin_name)
        if waypoint is None and quit_on_missing_waypoint:
            return None
        waypoints.append(get_closest_waypoint(graph, bin_name))
    
    return waypoints


def get_max_distance(graph):
    assert hasattr(graph, "distances"), "Graph lacks pre-computed distances dict!"
    distances = []
    for path_dict in graph.distances.values():
        for distance in path_dict.values():
            distances.append(distance)
    
    return max(distances)


def get_mean_distance(graph):
    assert hasattr(graph, "distances"), "Graph lacks pre-computed distances dict!"
    distances = []
    for path_dict in graph.distances.values():
        for distance in path_dict.values():
            distances.append(distance)
    
    return np.mean(distances)


def format_graph(graph):
    
    lr = max(list(graph))
    dims = lr[0] + 1, lr[1] + 1
    G = nx.Graph()
    G.dimensions = dims

    for k, _ in graph.nodes(data=True):
        coords = (k[0], k[1])
        G.add_node(str(coords), pos=tuple(coords), weight=1)
    
    nodes = list(G)
    
    for n in range(G.number_of_nodes()):
        if n%G.dimensions[1] > 0:
            G.add_edge(nodes[n],nodes[n-1])
        if n//G.dimensions[1]:
            G.add_edge(nodes[n],nodes[n-G.dimensions[1]])

    nx.set_node_attributes(G, "#AAB4D9", name="node_color")

    add_shortest_paths(G)
    add_distances(G)

    return G
