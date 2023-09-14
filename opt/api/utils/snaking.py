from opt.api.utils.general import chunk
from opt.graph import get_closest_waypoint_multi
from opt.algorithms.common import create_duplicate_node_multi



def snaking_start_end_constraints(fixed_start, fixed_end, unique_waypoints):

    if not (fixed_start or fixed_end):
        locations, start, end = unique_waypoints, None, None
    
    elif fixed_start and not fixed_end:
        locations, start, end = unique_waypoints[1:], unique_waypoints[0], None

    elif not fixed_start and fixed_end:
        locations, start, end = unique_waypoints[:-1], None, unique_waypoints[-1]

    elif fixed_start and fixed_end:
        locations, start, end = unique_waypoints[1:-1], unique_waypoints[0], unique_waypoints[-1]
    
    return locations, start, end


def get_pick_batches(pick_list, batch_size=32):
    return list(chunk(pick_list, batch_size))


def setup_pick_data(graph, assignment_data, use_bins):
    if use_bins:
        print("Using bin locations (getting nearest waypoints)")
    waypoints = get_closest_waypoint_multi(graph, assignment_data) if use_bins else assignment_data
    unique_waypoints = create_duplicate_node_multi(waypoints)
    node_to_bin_map = { unique_waypoints[idx]: assignment_data[idx] for idx in range(len(waypoints)) }

    return unique_waypoints, node_to_bin_map


def constrained_nodes_in_batch(constraints, batch):
    return[set([x for x in c if c in batch]) for c in constraints]
