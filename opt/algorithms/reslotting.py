import copy

import numpy as np
import networkx as nx

from opt.graph import get_shortest_path, get_distance, get_distance_multi, get_node_list
from opt.algorithms.common import convert_duplicate_node_multi
from opt.utils import to_str, conditional_int



def is_destination(node, suffix='_d.'):
    """
    Given encoding of sources and destinations as < num_items and >=
    num_items, respectively, return the category of a node
    """
    return type(node) in [str, np.str_] and suffix in node


def occupied(sources, destination):
    return destination in sources


def encode_destination_nodes(sources, destinations, suffix='_d.', prefix='_o.'):
    return [f'{prefix*occupied(sources, destinations[idx])}{destinations[idx]}{suffix}{sources[idx]}' for idx in range(len(destinations))]


def source_of(destination_node, suffix='_d.'):
    return conditional_int(destination_node.split(suffix)[1])


def corresponding_source(destination_node, suffix='_d.', prefix='_o.'):
    node = destination_node.split(suffix)[0]
    return conditional_int(node.split(prefix)[1]) if prefix in node else None


def corresponding_node(destination_node, suffix='_d.', prefix='_o.'):
    node = destination_node.split(suffix)[0]
    return conditional_int(node.split(prefix)[1]) if prefix in node else conditional_int(node)


def decode_destination_nodes(sequence, suffix='_d.', prefix='_o.'):
    return [corresponding_node(n, suffix, prefix) if is_destination(n, suffix) else n for n in sequence]


def swapped_out(state, idx, suffix, prefix):
    if idx == len(state) - 1:
        return False
    return state[idx+1] == corresponding_source(state[idx], suffix, prefix)
    

def constraints(S, num_slots, suffix='_d.', prefix='_o.'):
    """
    Check whether constraints on a proposed S are satisified:
      C1: Each destination node is preceded in S by its corresponding source node
      C2: Number of items in cart never exceeds its capacity
      C3: If a destination node is already occupied, occupant must come next in sequence
    """
    C1 = C2 = C3 = True
    sources_visited = set()
    destinations_visited = set()

    for idx, node in enumerate(S):
        if is_destination(node, suffix): 
            destinations_visited.add(node)
            C1 = source_of(node, suffix) in sources_visited
            if not C1: return False
            occupant = corresponding_source(node, suffix, prefix)
            C3 = (occupant is None) or (occupant in sources_visited) or swapped_out(S, idx, suffix, prefix)
            if not C3: return False
        else:
            sources_visited.add(node)
        
        C2 = 0 <= len(sources_visited) - len(destinations_visited) <= num_slots
        if not C2: return False
    
    return True


# TODO: Use tabular approach?
def reslotting_constraints(S, num_slots=1, suffix='_d.', prefix='_o.'):
    """
    Check whether constraints on a proposed S are satisified:
      C1: Each destination node is preceded in S by its corresponding source node
      C2: Number of items in cart never exceeds its capacity
      C3: Source item removed from destination before placing new item there
    """
    C1 = C2 = C3 = True
    sources_visited = set()
    destinations_visited = set()

    for idx, node in enumerate(S):
        if is_destination(node, suffix): 
            destinations_visited.add(node)
            
            C1 = C1 and (source_of(node, suffix) in sources_visited)
            occupant = corresponding_source(node, suffix, prefix)
            C3 = C3 and ((occupant is None) or (occupant in sources_visited) or swapped_out(S, idx, suffix, prefix))
        else:
            sources_visited.add(node)
        
        C2 = C2 and (0 <= len(sources_visited) - len(destinations_visited) <= num_slots)
    
    return C1, C2, C3


def reslotting_distance_constraints(S, start, end, graph, decoder, decoder_kwargs=None, suffix='_d.'):

    node_list = get_node_list(S, start, end)

    sources, destinations = [], []
    for node in node_list:
        if is_destination(node, suffix):
            destinations.append(node)
        else:
            sources.append(node)

    if decoder:
        decoder_kwargs = decoder_kwargs or dict()
        source_list = decoder(sources, **decoder_kwargs)
        destination_list = decoder(destinations, **decoder_kwargs)
    
    source_list = convert_duplicate_node_multi(source_list)
    destination_list = convert_duplicate_node_multi(destination_list)

    source_dest_distances = sum(
        [np.abs(node_list.index(x) - node_list.index(corresponding_source(x))) for x in destinations])
    
    return [get_distance_multi(graph, source_list) + get_distance_multi(graph, destination_list), 0.5*source_dest_distances]


def valid_initialization(sources, virtual_destinations, suffix='_d.', prefix='_o.'):
    # TODO : Use cart capacity to arrive at a potentially better initialization??
    sources_to_use = copy.deepcopy(sources)
    destinations_to_use = copy.deepcopy(virtual_destinations)
    initial_state = [sources_to_use.pop(0), destinations_to_use.pop(0)]

    while sources_to_use:
        occupant = corresponding_source(initial_state[-1], suffix, prefix)
        while occupant:
            if occupant in initial_state:
                break
            initial_state += [occupant, virtual_destinations[sources.index(occupant)]]
            sources_to_use.pop(sources_to_use.index(occupant))
            destinations_to_use.pop(destinations_to_use.index(virtual_destinations[sources.index(occupant)]))
            occupant = corresponding_source(initial_state[-1], suffix, prefix)
        if sources_to_use:
            next_in_line = sources_to_use.pop(0)
            initial_state += [next_in_line, virtual_destinations[sources.index(next_in_line)]]
    
    return initial_state


def unoptimized(graph, sources, destinations, start=None, end=None, use_bins=False, cumulative_distances=False):
    """
    Return the sequence of nodes, cost and full path for a fully unoptimized 
    solution to the reslot sequencing problem for a list of swaps [(A, B), (C, D), ...]:

    [A --> B --> A --> C --> D --> C --> ...]

    Input: graph, list of source node identifiers, (un-encoded) list of destination node identifiers;
        start, end nodes if applicable
    """

    previous = start if start is not None else sources[0]

    if use_bins:
        assert hasattr(graph, "bins"), "Graph has no `bins` attribute"
        path = [graph.bins[previous]["closest_waypoint"]]
    else:
        path = [previous]
    
    cost = 0
    if cumulative_distances: cumulative_cost = [0]

    for idx in range(1, len(sources)):
        source_node = sources[idx-1]
        destination_node = destinations[idx-1]
        next_source = sources[idx]

        path += get_shortest_path(graph, source_node, destination_node, use_bins=use_bins)[1:]
        cost += get_distance(graph, source_node, destination_node, use_bins=use_bins) if source_node != destination_node else 0
        if cumulative_distances: cumulative_cost.append(cost)

        path += get_shortest_path(graph, destination_node, next_source, use_bins=use_bins)[1:]
        cost += get_distance(graph, destination_node, next_source, use_bins=use_bins) if destination_node != next_source else 0
        if cumulative_distances: cumulative_cost.append(cost)
    
    path += get_shortest_path(graph, sources[len(sources) - 1], destinations[len(sources) - 1], use_bins=use_bins)[1:]
    cost += get_distance(graph, sources[len(sources) - 1], destinations[len(sources) - 1], use_bins=use_bins)
    if cumulative_distances: cumulative_cost.append(cost)

    if end is not None:
        path += get_shortest_path(graph, previous, end, use_bins=use_bins)[1:]
        cost += get_distance(graph, previous, end, use_bins=use_bins) if previous != end else 0

        if cumulative_distances: cumulative_cost.append(cost)

    virtual_destinations = encode_destination_nodes(sources, destinations)

    if use_bins:
        if start is not None:
            start = graph.bins[start]["closest_waypoint"]
        if end is not None:
            end = graph.bins[end]["closest_waypoint"]

    start = [] if start is None else [start]
    end = [] if end is None else [end]

    sequence = start + [node for item in zip(sources, virtual_destinations) for node in item] + end
    
    if cumulative_distances:
        cost = cumulative_cost
    return sequence, path, cost  


def sd_from_bin_columns(bin_A_list, bin_B_list, convert_tuples_to_str=True):
    """
    Given a list of swaps in [`bin_A`, `bin_B`] format, output the explicit list of `source`
    and `destination` bins, which is what the algo operates on (i.e. bin A1 is source and B1
    is the corresponding destination, and also B1 is source for destination A1)
    """
    sources = []
    destinations = []
    for idx in range(len(bin_A_list)):
        sources.append(bin_A_list[idx])
        sources.append(bin_B_list[idx])
        destinations.append(bin_B_list[idx])
        destinations.append(bin_A_list[idx])

    if type(sources[0])==tuple and convert_tuples_to_str:
        return to_str(sources), to_str(destinations)

    return sources, destinations


def bin_columns_from_sd(sources, destinations):
    """
    Reverse of `sd_from_bin_columns` - recovers original swap list (Bin A, Bin B columns) from sources and destinations.
    """
    binA_view1, binA_view2 = sources[::2], destinations[1::2]
    binB_view1, binB_view2 = sources[1::2], destinations[::2]

    assert (binA_view1 == binA_view2) and (binB_view1 == binB_view2), "Input data not in correct format; redundancy check failed!"
    return binA_view1, binB_view1


def swapped_state(graph, swap_columns, use_bins=True):
    """
    Given a graph and a series of swaps to carry out, return the "swapped" state of the graph, where
    location data from column A is exchanged with that for column B
    """
    G_swapped = copy.deepcopy(graph)

    locs_A, locs_B = swap_columns[0], swap_columns[1]

    if use_bins:
        for (a, b) in zip(locs_A, locs_B):
            G_swapped.bins[a] = graph.bins[b]
            G_swapped.bins[b] = graph.bins[a]
    else:
        sources, destinations = sd_from_bin_columns(*swap_columns)
        swap_dict = {k: v for k, v in zip(sources, destinations)}
        G_swapped = nx.relabel_nodes(G_swapped, swap_dict)
    
    return G_swapped
