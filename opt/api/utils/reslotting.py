from opt.algorithms.reslotting import is_destination, encode_destination_nodes, decode_destination_nodes, sd_from_bin_columns
from opt.algorithms.common import create_duplicate_node_multi, convert_duplicate_node_multi
from opt.graph import get_mean_distance, get_max_distance, get_closest_waypoint_multi, get_distance_multi
from opt.model.sequence import Sequence
from opt.api.utils.general import chunk



def bin_role_map(nodes, source_bins, destination_bins, source_nodes, encoded_destination_nodes, encoder):
    """
    Map a list of role-encoded nodes to the corresponding role-encoded bin names, 
    given the source and destination bins, source nodes, encoded destination nodes, and encoding scheme
    """
    encoded_destination_bins = encoder(source_bins, destination_bins)
    destination_map = {encoded_destination_nodes[idx]: encoded_destination_bins[idx] for idx in range(len(encoded_destination_nodes))}
    source_map = {source_nodes[idx]: source_bins[idx] for idx in range(len(source_nodes))}

    return [destination_map[n] if is_destination(n) else source_map[n] for n in nodes]
   


def start_end_constraints(fixed_start, fixed_end, source_waypoints, destination_waypoints):
    if not (fixed_start or fixed_end):
        locations, start, end = source_waypoints + destination_waypoints, None, None
    
    elif fixed_start and not fixed_end:
        locations, start, end = source_waypoints[1:] + destination_waypoints, source_waypoints[0], None

    elif not fixed_start and fixed_end:
        locations, start, end = source_waypoints + destination_waypoints[:-1], None, destination_waypoints[-1]

    elif fixed_start and fixed_end:
        locations, start, end = source_waypoints[1:] + destination_waypoints[:-1], source_waypoints[0], destination_waypoints[-1]

    return locations, start, end


def reslotting_results(graph, sequence, sources, destinations, source_waypoints, encoded_destination_nodes, cart_capacity, use_bins, cumulative_distances):
    if use_bins:
        bin_sequence = bin_role_map(
            sequence, 
            sources, 
            destinations,
            source_waypoints,
            encoded_destination_nodes, 
            encode_destination_nodes
        )
        decoded_sequence = decode_destination_nodes(bin_sequence)
        sequence = Sequence(bin_sequence, graph, use_bins, cart_capacity=cart_capacity)
    else:
        decoded_sequence = decode_destination_nodes(sequence)
        sequence = Sequence(sequence, graph, use_bins,cart_capacity=cart_capacity)
    
    decoded_sequence = convert_duplicate_node_multi(decoded_sequence)

    return sequence, get_distance_multi(graph, decoded_sequence, use_bins=use_bins, cumulative_distance=cumulative_distances)


def setup_reslotting_data(graph, sources, destinations, use_bins, num_swaps=None):
    # Get waypoints corresponding to bins if appliable
    source_waypoints = get_closest_waypoint_multi(graph, sources, quit_on_missing_waypoint=False)[:num_swaps] \
        if use_bins else sources[:num_swaps]
    destination_waypoints = get_closest_waypoint_multi(graph, destinations, quit_on_missing_waypoint=False)[:num_swaps] \
        if use_bins else destinations[:num_swaps]

    # In case of duplicates, yet another layer of encoding
    source_waypoints = create_duplicate_node_multi(source_waypoints, mode="source")
    destination_waypoints = create_duplicate_node_multi(destination_waypoints, mode="destination")

    if not source_waypoints or not destination_waypoints:
        print("No waypoints for graph could be found corresponding to locations in swap list!")
        return None, None, None

    encoded_destination_nodes = encode_destination_nodes(source_waypoints, destination_waypoints)

    return source_waypoints, destination_waypoints, encoded_destination_nodes


def setup_reslotting_weights(graph, constraint_weight_base, constraint_weight):
    if constraint_weight_base:
        distance_summary = get_mean_distance(graph) if constraint_weight_base == "mean" else get_max_distance(graph)
    else:
        distance_summary = 1
    return constraint_weight*distance_summary


def get_swap_batches(bin_A_list, bin_B_list, batch_size=32, return_sources_and_destinations=True):
    bin_A_batches = list(chunk(bin_A_list, batch_size))
    bin_B_batches = list(chunk(bin_B_list, batch_size))

    if return_sources_and_destinations: 
        return [sd_from_bin_columns(bin_A_batches[idx], bin_B_batches[idx]) for idx in range(len(bin_A_batches))]
    return [bin_A_batches, bin_B_batches]
    