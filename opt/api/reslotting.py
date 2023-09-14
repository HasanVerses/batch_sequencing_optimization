import os

import networkx as nx
import numpy as np

from opt.api.core import solve_tsp
from opt.model.sequence import Sequence
from opt.algorithms.common import null_result
from opt.algorithms.reslotting import (
    unoptimized, 
    valid_initialization, 
    decode_destination_nodes, 
    reslotting_constraints,
    reslotting_distance_constraints,
    swapped_state
)
from opt.api.utils.reslotting import (
    reslotting_results, 
    setup_reslotting_data, 
    setup_reslotting_weights, 
    start_end_constraints,
    get_swap_batches
)
from opt.api.utils.general import format_kwargs, check_algorithm, get_single_image_path
from opt.graph import get_closest_waypoint_multi, format_graph
from opt.io.local import (
    load_reslotting_data, 
    parse_reslotting_input, 
    parse_domain_input, 
    save_reslotting_data, 
    save_reslotting_data_batch
)
from opt.api.snaking import analyze_picks
from opt.visual.visual import plot_graph, animated_plot



def swaps_from_csv(reslotting_data_path, num_swaps=None, save_result_path=None, return_sources_and_destinations=False):
    """
    Load reslotting ("swap list") data stored in a .csv file for use with reslotting sequencing/analysis algorithms.

    Parameters: 
        `reslotting_data_path`: str: Relative path to .csv file containing swaps to execeute

        `num_swaps`: int: Specifies how many swaps to load from the datafile

        `save_result_path`: str: If supplied, result will be stored as a numpy array at this location.
    
    Returns:
        
        [ [ bin_name: str per Pick task ] per Pick assignment ]

        Optionally saves result as a .npy file.

    """
    assert os.path.exists(reslotting_data_path), f"Pick data at {reslotting_data_path} could not be found!"
    
    print("Reading swap data...")
    swaps = load_reslotting_data(reslotting_data_path, num_swaps, return_sd=return_sources_and_destinations)

    if save_result_path:
        print(f"Saving result as a numpy array at {save_result_path}...")
        np.save(save_result_path, swaps)
    print("Done.")

    return swaps


def load_csv(swap_data_path, **kwargs):
    """Convenient alias for `reslotting_swaps_from_csv`"""
    return swaps_from_csv(swap_data_path, **kwargs)


def load_npy(reslotting_data_path):
    """Load saved numpy array"""
    return np.load(reslotting_data_path, allow_pickle=True)


def analyze_swaps(
    domain_id_or_graph,
    swap_data_or_path,
    num_swaps=None,
    start=None,
    end=None,
    print_results=False,
    display_image=False,
    output_df=True,
    save_csv=True,
    cumulative_distances=True,
    save_image_path="",
    save_animation_path="",
    use_bins=True,
    **kwargs
):
    """
    Return the sequence of nodes, cost and full path for a fully unoptimized 
    solution to the reslot sequencing problem for a list of swaps [(A, B), (C, D), ...]:

    [A --> B --> A --> C --> D --> C --> ...]

    Parameters:
        `domain_id_or_graph`: str or nx.classes.graph.Graph: One of:
            * Pre-loaded domain graph
            * String identifying a domain graph to load/download:
                o Domain abbreviation (e.g. "LI")
                o Domain identifier (e.g. "dom.nri.us.li")
                o Domain SWID (e.g. "e20deb3b-2144-4061-b9d7-65d87994c64b")
        
        `swap_data_or_path`: str or List[List[str]]: One of:
            * Swap list data (list of "sources" (bin A) and "destinations" (bin B))
            * Path to swap list data
                o .csv (default/expected)
                o .npy (quicker to load; see save option in `pick_data_from_csv`)
        
        `start` / `end`: node identifiers representing arbitrary start/end locations (distinct from the
            swap locations specified in the data)
        
        `sequence_format`: str: Sequences of moves an be returned as `English` instructions,
            `decoded` node IDs, or `raw` 'virtual node' designations (used internally by the algo)

        `save_image_path`: str: If specified, save a plot of the assignment to this location
        `save_animation_path`: str: If specified, save an animation of the assignment to this location
    
    Returns:
        Pandas dataframe if `output_csv` is True, otherwise:

        `distance`: Distance of path, taking into account waypoint <--> bin distances
        `sequence`: Sequence of nodes to visit / (pick, place) moves to make
        `path`: List of nodes in the graph representing the path taken

    """
    G = parse_domain_input(domain_id_or_graph)

    sources, destinations = parse_reslotting_input(swap_data_or_path, return_sources_and_destinations=True)
    if use_bins:
        source_waypoints = get_closest_waypoint_multi(G, sources, quit_on_missing_waypoint=False)[:num_swaps]
        destination_waypoints = get_closest_waypoint_multi(G, destinations, quit_on_missing_waypoint=False)[:num_swaps]
    else:
        source_waypoints, destination_waypoints = sources, destinations

    if not source_waypoints or not destination_waypoints:
        print("No waypoints for graph could be found corresponding to locations in swap list!")
        return None
    
    raw_sequence, path, distance = unoptimized(G, sources, destinations, start, end, use_bins=use_bins, cumulative_distances=cumulative_distances)
    sequence = Sequence(raw_sequence, G, use_bins, cart_capacity=1)

    if print_results:
        print("Analyzing swap data...")
        print("Distance: ", distance)
        print("Sequence: ", sequence)
        print("Path: ", path)

    plot_kwargs, anim_kwargs, kwargs = format_kwargs(kwargs, save_image_path, display_image, save_animation_path)

    if display_image or save_image_path:
        print("Plotting graph...")
        plot_graph(G, sequence=sequence, **plot_kwargs)

    if save_animation_path:
        print("Saving animation...")
        animated_plot(G, sequence=sequence, **anim_kwargs)

    if output_df:
        print("Saving .csv...")
        domain_str = ("DC" if use_bins else "graph") if type(domain_id_or_graph) != str else domain_id_or_graph
        distance_data = distance if cumulative_distances else None
        return save_reslotting_data(
            sequence.english, 
            None, 
            1, 
            "swaps", 
            cumulative_distances, 
            distance_data, 
            domain_str, 
            "", 
            save_it=save_csv
        )
    return distance, sequence, path


def optimize_swaps(
    domain_id_or_graph, 
    swap_data_or_path,
    cart_capacity=8,
    num_swaps=None,
    batch_size=None,
    constraint_weight=3.,
    constraint_weight_base="mean",
    naturalness_constraint_weight=0.2,
    fixed_start=False,  # Note: NOT FULLY TESTED; recommend against changing
    fixed_end=False,    # Note: NOT FULLY TESTED; recommend against changing
    algorithm="annealing",
    valid_initial_state=True,
    random_seed=None,
    print_results=False,
    display_image=False,
    output_csv_format="move_list",
    cumulative_distances=True,
    save_image_path="",
    save_animation_path="",
    use_bins=True,
    compare_to_baseline=True,
    **kwargs
):
    """
    Optimize a set of picks in a pick assignment for distance, given a domain and pick assignment data.

    Parameters:
        Same as `analyze_swaps` above except where specified.

        `cart_capacity`: int: Specifies how many items can be carried simultaneously while carrying out item
        reslotting. In the special case of cart_capacity=1, all the algorithm can do is optimize the order of the
        (A -> B -> A) swaps specified in the input data, since only one can be carried at a time.

        `num_swaps`: int: Maximum number of swaps from the list to carry out.

        `batch_size`: int: Number of swaps per batch -- if supplied, splits input into batches and returns a list of
          results for each, or a dataframe with all results collected and a 'Batch' index

        `constraint_weight`: int: specifies multiplier for base constraint weight (see next arg)
        `constraint_weight_base`: str: use `mean` or `max` of shortest paths in graphs; 
            multiplied by `constraint_weight`; if None, set to 1.

        `fixed_start`: bool: If True, only search for paths beginning with the first element of the input sequence.
        `fixed_end`: bool: If True, only search for paths ending with the last element of the input sequence.

        `algorithm`: str: specifies which algorithm to use:
            * "annealing": Simulated annealing (see `algorithms.annealing.anneal` definition for kwargs)
            * "threshold": Deterministic simulated annealing (see `algorithms.annealing.deterministic_anneal`)
            * "genetic": Genetic algorithm (see `algorithms.genetic.genetic` definition for kwargs)
            * "genetic x": Genetic algorithm with crossover (see `algorithms.genetic.genetic_crossover`)
            * "naive": Naive / brute-force solution (see `algorithms.naive.naive_tsp` definition)
        
        `valid_initial_state`: bool: Specifies whether to initialize approximate algorithms with a
        state that meets specified constraints. 

        `random_seed`: int: specifies random seed to use for numpy.

        `output_csv_format`: str: output a .csv file of the optimized sequence in the following 
          formats:
            If `move_list`: [move_type, from_location, (to_location), (distance)] - Lists type of move ("Pick", "Place"), 
            `from_location`: source location of item being picked or place; `to_location`: destination location 
            of item being placed; `distance`: cumulative distances if `cumulative_distances` flag is set

            If `move_descriptions`: [move_description, (distance)] - List of instructions in the form "Pick item from location X" 
            or "Place item from location X in location Y" needed to carry out the input swaps.

            If `swaps`: optimized sequence of swaps in the same format as the input.

            If None: do not save a .csv file.
        
        `cumulative_distances`: bool: toggles whether to return (and save, if applicable) the cumulative
        distance travelled at each step of the swap/move list.

        **kwargs: Additional arguments to pass on to optimization function.

    Returns:
        `distance` = The length of the path (including waypoint <--> bin distances)
        `sequence` = Optimized sequence of bin locations
        `path` = Sequence of node identifiers representing the path through the graph found by the algorithm 
            (including all intermediate nodes traversed)
        
    """
    check_algorithm(algorithm)
    assert constraint_weight_base in ["max", "mean", None], "Please set `constraint_weight_base` to `max` or `mean`"
    assert output_csv_format in ["move_list", "move_descriptions", "swaps", None], "Valid output .csv formats are `from_to`, `move_list` and `swaps`"

    G = parse_domain_input(domain_id_or_graph)
    assert nx.is_connected(G), "Graph is not connected; problem may be ill-posed"

    if random_seed:
        np.random.seed(random_seed)

    constraint_penalty = setup_reslotting_weights(G, constraint_weight_base, constraint_weight)

    plot_kwargs, anim_kwargs, kwargs = format_kwargs(kwargs, save_image_path, display_image, save_animation_path)

    bin_A_list, bin_B_list = parse_reslotting_input(swap_data_or_path, num_swaps)

    # Parse into batches, if `batch_size` supplied
    num_tasks = len(bin_A_list)
    batch_size = batch_size or num_tasks
    batches = get_swap_batches(bin_A_list, bin_B_list, batch_size)
    num_batches = len(batches)
    extra = num_tasks%batch_size
    if num_batches > 1:
        extra_str = f" plus remainder batch containing {extra} swaps" if extra else ""
        print(f"Split input data into {num_batches - (extra != 0)} batch(es) containing {batch_size} swaps each{extra_str}.")

    # Loop over batches
    sequences, paths, distances = [], [], []
    for batch_idx, batch in enumerate(batches):
        if num_batches > 1:
            print(f"Processing batch {batch_idx}...")
        sources, destinations = batch[0], batch[1]

        source_waypoints, destination_waypoints, encoded_destination_nodes = setup_reslotting_data(
            G, 
            sources, 
            destinations, 
            use_bins
        )
        if not source_waypoints or not destination_waypoints:
            print("No waypoints for graph could be found corresponding to locations in swap list!")
            if num_batches==1:
                return null_result()
            else:
                sequences.append(None)
                paths.append(None)
                distances.append(np.nan)
                continue

        if valid_initial_state is not False and algorithm != "naive":
            kwargs["initial_state"] = valid_initialization(source_waypoints, encoded_destination_nodes)

        locations, start, end = start_end_constraints(fixed_start, fixed_end, source_waypoints, destination_waypoints)

        print(f"Using {algorithm} algorithm to search for an optimized reslotting path")
        sequence, path, _ = solve_tsp(
            G, 
            locations, 
            start, 
            end,
            algorithm=algorithm,
            decoder=decode_destination_nodes, 
            constraint_fn=reslotting_constraints,
            constraint_penalty=constraint_penalty,
            constraint_fn_kwargs={"num_slots": cart_capacity},
            cost_fn=reslotting_distance_constraints,
            cost_penalty=naturalness_constraint_weight,
            cost_fn_kwargs={"start": start, "end": end, "graph": G, "decoder": decode_destination_nodes},
            **kwargs
        )

        if sequence is None:
            print("No valid result obtained with given parameters!")
            if num_batches == 1:
                return null_result()
            sequences.append(None)
            paths.append(None)
            distances.append(np.nan)
            continue
        
        sequence, distance = reslotting_results(
            G, 
            sequence,
            sources, 
            destinations, 
            source_waypoints, 
            encoded_destination_nodes, 
            cart_capacity, 
            use_bins, 
            cumulative_distances
        )

        sequences.append(sequence)
        paths.append(path)
        distances.append(distance)

        if print_results:
            if cumulative_distances:
                print("Distance: ", distance[-1])
            else:
                print("Distance: ", distance)
            print("Sequence: ", sequence)
            print("Path: ", path)
    
        if compare_to_baseline:
            _, _, naive_distance = unoptimized(G, batch[0], batch[1], start, end, use_bins=use_bins, cumulative_distances=False)
            print("Naive distance: ", naive_distance)
            print("Savings: ", naive_distance - (distance[-1] if cumulative_distances else distance))
            sequence.add_baseline(naive_distance) #TODO : Make this less hacky

        if display_image or save_image_path or save_animation_path:
            if display_image or save_image_path:
                if save_image_path:
                    plot_kwargs['save_image_path'] = \
                        get_single_image_path(save_image_path, f'batch_{str(batch_idx).zfill(4)}') if num_batches > 1 else save_image_path #f'batch_{str(batch_idx).zfill(4)}_' + plot_kwargs['save_image_path']
                print("Plotting graph...")
                plot_graph(G, sequence=sequence, **plot_kwargs)
        
            if save_animation_path:
                anim_kwargs['output_path'] = \
                    get_single_image_path(save_animation_path, f'batch_{str(batch_idx).zfill(4)}') if num_batches > 1 else save_animation_path
                print("Saving animation...")
                animated_plot(G, sequence=sequence, **anim_kwargs)

    domain_str = ("DC" if use_bins else "graph") if type(domain_id_or_graph) != str else domain_id_or_graph

    if len(sequences) == 1:
        distance = distances[0]
        sequence = sequences[0]
        path = paths[0]

        if output_csv_format is not None:
            distance_data = distance if cumulative_distances else None
            return save_reslotting_data(
                sequence.english, 
                {s: d for s, d in zip(sources, destinations)}, 
                cart_capacity,
                output_csv_format, 
                cumulative_distances,
                distance_data, 
                domain_str,
                f'{algorithm}_optimized'
            )
    else:  # Multiple batches
        if output_csv_format is not None:
            distance_data = distances if cumulative_distances else None
            return save_reslotting_data_batch(
                sequences, 
                batches, 
                cart_capacity,
                output_csv_format, 
                cumulative_distances,
                distance_data, 
                domain_str,
                f'{algorithm}_optimized'
            )        
        return distances, sequences, paths
    
    return distance, sequence, path


def analyze_swaps_grid(graph, swap_data_or_path, convert_graph=True, node_labels_on=True, **kwargs):
    """Alias for `analyze_swaps` with `use_bins` set to False (appropriate for grid graphs)"""
    if convert_graph:
        graph = format_graph(graph)
    kwargs["node_labels_on"] = node_labels_on
    kwargs["use_bins"] = False

    results = analyze_swaps(graph, swap_data_or_path, **kwargs)
    if convert_graph:
        return graph, *results
    return results

def optimize_swaps_grid(graph, swap_data_or_path, convert_graph=True, node_labels_on=True, **kwargs):
    """Alias for `optimize_swaps` with `use_bins` set to False (appropriate for grid graphs)"""
    if convert_graph:
        graph = format_graph(graph)
    kwargs["node_labels_on"] = node_labels_on
    kwargs["use_bins"] = False
    
    results = optimize_swaps(graph, swap_data_or_path, **kwargs)
    if convert_graph:
        return graph, *results
    return results


def pre_post_reslotting(domain_id_or_graph, swap_data_or_path, pick_data=None, use_bins=True, **kwargs):
    """
    Given a graph, swaps in the format [bin column A, bin column B], and (optional) pick sequences,
    compute analysis of picks pre- and post-swapping.

    Uses column A in the swap list as the pick data by default.
    kwargs are passed to `analyze_picks`.
    """
    G = parse_domain_input(domain_id_or_graph)
    swap_data = parse_reslotting_input(swap_data_or_path, return_sources_and_destinations=False)
    G2 = swapped_state(G, swap_data, use_bins)

    if pick_data is None:
        pick_data = swap_data[0] # Use column A from swap list by default

    pre_distance, pre_path = analyze_picks(G, pick_data, **kwargs)
    post_distance, post_path = analyze_picks(G2, pick_data, **kwargs)

    return pre_distance, pre_path, post_distance, post_path
