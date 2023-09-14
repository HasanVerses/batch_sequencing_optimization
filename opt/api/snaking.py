import os

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from opt.api.core import solve_tsp
from opt.model.sequence import Sequence
from opt.algorithms.common import (
    convert_duplicate_node_multi, 
    valid_initial_state,
    constraint_waypoints
)
from opt.utils import to_str
from opt.api.utils.snaking import snaking_start_end_constraints, get_pick_batches, setup_pick_data, constrained_nodes_in_batch
from opt.api.utils.general import get_single_image_path, check_algorithm, format_kwargs
from opt.graph import get_shortest_path_multi, get_distance_multi, format_graph
from opt.io.local import bins_from_pick_data, parse_domain_input, parse_assignment_input, filter_assignments, save_pick_data, save_pick_data_batch
from opt.graph import get_closest_waypoint_multi
from opt.visual.visual import plot_graph, animated_plot, add_labels



def pick_data_from_csv(pick_data_path, num_assignments=None, filter_short_assignments=False, save_result_path=None):
    """
    Load pick data stored in a .csv file and format for use with sequencing/analysis algorithms.

    Parameters: 
        `pick_data_path`: str: Relative path to .csv file containing pick data (format: "path/to/data.csv")

        `num_assignments`: int: Specifies how many assignments from the total pick data to return

        `filter_short_assignments`: bool: Toggle whether to toss out assignments with fewer than 3 pick locations
            (which cannot be optimized)

        `save_result_path`: str: If supplied, result will be stored as a numpy array at this location.
    
    Returns:
        [ [ bin_name: str per Pick task ] per Pick assignment ]

        Optionally saves result as a .npy file.

    """
    assert os.path.exists(pick_data_path), f"Pick data at {pick_data_path} could not be found!"

    print("Reading pick data...")
    pick_locations = bins_from_pick_data(pick_data_path)

    if filter_short_assignments:
        print("Filtering short assignments...")
        pick_locations = filter_assignments(pick_locations)

    if save_result_path:
        print(f"Saving result as a numpy array at {save_result_path}...")
        np.save(save_result_path, pick_locations)
    print("Done.")

    return pick_locations[:num_assignments]


def load_csv(pick_data_path, **kwargs):
    """Convenient alias for `pick_data_from_csv`"""
    return pick_data_from_csv(pick_data_path, **kwargs)


def load_npy(pick_data_path):
    """Load saved numpy array"""
    return np.load(pick_data_path, allow_pickle=True)


def analyze_picks(
    domain_id_or_graph,
    pick_data_or_path,
    cumulative_distances=True,
    batch_size=None,
    num_picks=None,
    output_df=True,
    save_csv=True,
    print_results=False,
    display_image=False,
    save_image_path="",
    save_animation_path="",
    use_bins=True,
    **kwargs
):
    """
    Return distance of the shortest path compatible with an input sequence of picks representing one assignment.
    Also returns the path itself.

    Parameters:
        `domain_id_or_graph`: str or nx.classes.graph.Graph: One of:
            * Pre-loaded domain graph
            * String identifying a domain graph to load/download:
                o Domain abbreviation (e.g. "LI")
                o Domain identifier (e.g. "dom.nri.us.li")
                o Domain SWID (e.g. "e20deb3b-2144-4061-b9d7-65d87994c64b")
        
        `assignment_data`: List[str]: bin location data representing a pick assignment

        `save_image_path`: str: If specified, save a plot of the assignment to this location
        `save_animation_path`: str: If specified, save an animation of the assignment to this location

        kwargs are parsed and passed on to print/animation functions as appropriate
    
    Returns:
        `distance`: Distance of path, taking into account waypoint <--> bin distances
        `path`: List of nodes in the graph representing the path taken

    """
    G = parse_domain_input(domain_id_or_graph)

    assignment_data = parse_assignment_input(pick_data_or_path, group_by_assignment=False)[:num_picks]

    if use_bins:
        waypoints = get_closest_waypoint_multi(G, assignment_data)
        if not waypoints:
            return None, None
    else:
        waypoints = assignment_data

    distance = get_distance_multi(G, assignment_data, use_bins=use_bins, cumulative_distance=cumulative_distances)
    path = get_shortest_path_multi(G, assignment_data, use_bins=use_bins)

    if print_results:
        print("Analyzing pick data...")
        print("Distance: ", distance)
        print("Path: ", path)

    plot_kwargs, anim_kwargs, _ = format_kwargs(kwargs, save_image_path, display_image, save_animation_path)

    if display_image or save_image_path or save_animation_path:
        sequence = Sequence(assignment_data, G, use_bins)

    if display_image or save_image_path:
        if display_image and print_results:
            print("Plotting graph...")
        if save_image_path and print_results:
            print("Saving plot...")
        if use_bins:
            plot_graph(G, sequence=sequence, **plot_kwargs)
        else:
            plot_graph(G, sequence=waypoints, **plot_kwargs)

    if save_animation_path:
        print("Saving animation...")
        if use_bins:
            animated_plot(G, sequence=sequence, **anim_kwargs)
        else:
            animated_plot(G, sequence=waypoints, **anim_kwargs)
    
    if output_df:
        domain_str = ("DC" if use_bins else "graph") if type(domain_id_or_graph) != str else domain_id_or_graph
        return save_pick_data(assignment_data, cumulative_distances, distance, save_str=domain_str, save_it=save_csv)

    return distance, path


def analyze_assignments(
    domain_id_or_graph,
    assignment_data_or_path,
    cumulative_distances=True,
    random_assignments=False,
    assignment_length_range=None,
    output_df=False,
    save_csv=False,
    print_results=False,
    display_image=False,
    save_image_path="",
    save_animation_path="",
    num_assignments=None,
    use_bins=True,
    **extra_kwargs
):
    """
    Return shortest path and distance data for multiple pick assignments.
    Same as `analyze_pick_assignment` except as noted:

    Parameters:
        `assignment_data_or_path`: str or List[List[str]]: One of:
            * Assignment data (list of lists of strings representing bins, one list per assignment)
            * Path to assignment data
                o .csv (default/expected)
                o .npy (quicker to load; see save option in `pick_data_from_csv`)
        
        `random_assignments`: bool: Randomly permute input assignment list. Can be used together with `num_assignments`
          and `assignment_length_range` to get a random sample of assignments of a given length
        `assignment_length_range`: Tuple[int, int]: consider only assignments of length between min and max values (inclusive)
        `num_assignments`: int: Only process first `num_assignments` assignments

    
    Output:
        `distances`: List of distances per assignment
        `paths`: List of shortest paths (compatible with input sequences) per assignment
    
    """
    G = parse_domain_input(domain_id_or_graph)
    assert nx.is_connected(G), "Graph is not connected; problem may be ill-posed"    
    assignment_data = parse_assignment_input(assignment_data_or_path, random_assignments, assignment_length_range, num_assignments)
    paths = dict()
    distances = dict()

    for idx, assignment in enumerate(assignment_data):
        print(f"Processing assignment {idx}")
        single_image_path = get_single_image_path(save_image_path, idx)
        single_animation_path = get_single_image_path(save_animation_path, idx, 'gif')
        distance, path = analyze_picks(
            G, 
            assignment,
            cumulative_distances,
            None,
            None,
            output_df,
            save_csv, 
            print_results, 
            display_image, 
            single_image_path, 
            single_animation_path,
            use_bins,
            **extra_kwargs
        )
        if not distance:
            print("Relevant bin data could not be located; skipping assignment.")
            continue

        paths[idx] = path
        distances[idx] = distance
    
    if output_df:
        # TODO : Implement
        pass
    
    return distances, paths
        

def optimize_picks(
    domain_id_or_graph, 
    pick_data_or_path,
    fixed_start=False, 
    fixed_end=False, 
    algorithm="annealing", 
    random_seed=None,
    cumulative_distances=True,
    batch_size=None,
    num_picks=None,
    output_df=True,
    save_csv=True,
    print_results=False,
    display_image=False,
    save_image_path="",
    save_animation_path="",
    use_bins=True,
    **kwargs
):
    """
    Optimize a set of picks in a pick assignment for distance, given a domain and pick assignment data.

    Parameters:
        Same as `analyze_picks` above except where specified.

        `fixed_start`: bool: If True, only search for paths beginning with the first element of the input sequence.
        `fixed_end`: bool: If True, only search for paths ending with the last element of the input sequence.

        `algorithm`: str [default: "annealing"]: specifies which algorithm to use:
            * "annealing": Simulated annealing (see `algorithms.annealing.anneal` definition for kwargs)
            * "genetic": Genetic algorithm (see `algorithms.genetic.genetic` definition for kwargs)
            * "genetic x": Genetic algorithm with crossover (see `algorithms.genetic.genetic_crossover`)
            * "naive": Naive / brute-force solution (see `algorithms.naive.naive_tsp` definition)

        `random_seed`: int [default: None] specifying random seed to use for numpy.

        **kwargs: Additional arguments to pass on to optimization function.

    Returns:
        `distance` = The length of the path (including waypoint <--> bin distances)
        `sequence` = Optimized sequence of bin locations
        `path` = Sequence of node identifiers representing the path through the graph found by the algorithm 
            (including all intermediate nodes traversed)
        
    """
    assignment_data = parse_assignment_input(pick_data_or_path, group_by_assignment=False)[:num_picks]

    if len(assignment_data) < 3:
        print("Fewer than 3 locations in input data; skipping assignment (can't be optimized).")
        return None, None, None

    check_algorithm(algorithm)
    G = parse_domain_input(domain_id_or_graph)
    assert nx.is_connected(G), "Graph is not connected; problem may be ill-posed"

    if random_seed:
        np.random.seed(random_seed)
    
    distance_baseline = kwargs['baseline_distance'] if 'baseline_distance' in kwargs else None #TODO : Make this less hacky
    plot_kwargs, anim_kwargs, kwargs = format_kwargs(kwargs, save_image_path, display_image, save_animation_path)
        
    # Parse into batches, if `batch_size` supplied
    num_tasks = len(assignment_data)
    batch_size = batch_size or num_tasks
    batches = get_pick_batches(assignment_data, batch_size)
    num_batches = len(batches)
    extra = num_tasks%batch_size
    if num_batches > 1:
        extra_str = f" plus remainder batch of size {extra}" if extra else ""
        print(f"Split input data into {num_batches - (extra != 0)} batch(es) of size {batch_size}{extra_str}.")

    # Loop over batches    
    sequences, paths, distances = [], [], []
    for batch_idx, batch in enumerate(batches):
        if num_batches > 1:
            print(f"Processing batch {batch_idx}...")

        unique_waypoints, node_to_bin_map = setup_pick_data(G, batch, use_bins)
        locations, start, end = snaking_start_end_constraints(fixed_start, fixed_end, unique_waypoints)

        if "constraint" in kwargs and algorithm != "naive":
            batch_constrained_nodes = constrained_nodes_in_batch(kwargs["constraint"], batch) if num_batches > 1 else kwargs["constraint"]
            mapped_constraint = (constraint_waypoints(G, batch_constrained_nodes) if use_bins else batch_constrained_nodes)
            initial_state = valid_initial_state(unique_waypoints, mapped_constraint)
            kwargs |= {"initial_state": initial_state, "constraint": mapped_constraint}

        print(f"Using {algorithm} algorithm to search for an optimized path")
        sequence, path, distance = solve_tsp(
            G, 
            locations, 
            start, 
            end, 
            algorithm=algorithm, 
            **kwargs
        )
        mapped_sequence = convert_duplicate_node_multi([node_to_bin_map[n] for n in sequence])
        distance = get_distance_multi(G, mapped_sequence, use_bins=use_bins, cumulative_distance=cumulative_distances)

        sequences.append(mapped_sequence)
        paths.append(path)
        distances.append(distance)

        if print_results:
            print("Distance: ", distance)
            print("Sequence: ", mapped_sequence)
            print("Path: ", path)

        viz_sequence = Sequence(mapped_sequence, G, use_bins)
        if distance_baseline is not None:
            viz_sequence.add_baseline(distance_baseline)

        if display_image or save_image_path:
            if save_image_path:
                plot_kwargs['save_image_path'] = \
                    get_single_image_path(save_image_path, f'batch_{str(batch_idx).zfill(4)}') if num_batches > 1 else save_image_path
            print("Plotting graph...")
            plot_graph(G, sequence=viz_sequence, **plot_kwargs)

        if save_animation_path:
            anim_kwargs['output_path'] = \
                get_single_image_path(save_animation_path, f'batch_{str(batch_idx).zfill(4)}') if num_batches > 1 else save_animation_path
            print("Saving animation...")
            animated_plot(G, sequence=viz_sequence, **anim_kwargs)

    if len(sequences) == 1:
        distance = distances[0]
        sequence = sequences[0]
        path = paths[0]

        if output_df:
            distance_data = distance if cumulative_distances else None
            domain_str = ("DC" if use_bins else "graph") if type(domain_id_or_graph) != str else domain_id_or_graph
            return save_pick_data(sequence, cumulative_distances, distance_data, save_str=domain_str, save_it=save_csv)
    
    else: # Multiple batches
        if output_df:
            distance_data = distances if cumulative_distances else None
            domain_str = ("DC" if use_bins else "graph") if type(domain_id_or_graph) != str else domain_id_or_graph
            return save_pick_data_batch(sequences, cumulative_distances, distance_data, save_str=domain_str, save_it=save_csv)

        return distances, sequences, paths

    return distance, sequence, path


def optimize_assignments(
    domain_id_or_graph, 
    assignment_data_or_path,
    fixed_start=True, 
    fixed_end=True, 
    algorithm="annealing", 
    random_seed=None, 
    random_assignments=False,
    assignment_length_range=None,
    cumulative_distances=True,
    output_df=False,
    save_csv=False,
    print_results=False,
    display_image=False,
    save_image_path="",
    save_animation_path="",
    num_assignments=None,
    use_bins=True,
    **kwargs
):
    """
    Optimize a set of pick assignments, given domain ID and multiple pick assignment data.

    Parameters:
        Same as `optimize_pick_assignment`, essentially.
        For plots, multiple plots/animations will be generated and saved with consecutive indices.

    Returns:
        Lists of results as in the single-assignment case
        
    """
    G = parse_domain_input(domain_id_or_graph)
    assert nx.is_connected(G), "Graph is not connected; problem may be ill-posed"
    assignment_data = parse_assignment_input(assignment_data_or_path, random_assignments, assignment_length_range, num_assignments)

    if random_seed:
        np.random.seed(random_seed)
    
    distances = []
    sequences = []
    paths = []

    for idx, assignment in enumerate(assignment_data):
        single_image_path = get_single_image_path(save_image_path, idx)
        single_animation_path = get_single_image_path(save_animation_path, idx, extension='gif')
        distance, sequence, path = optimize_picks(
            G, 
            assignment, 
            fixed_start=fixed_start, 
            fixed_end=fixed_end, 
            algorithm=algorithm,
            cumulative_distances=cumulative_distances,
            output_df=output_df,
            save_csv=save_csv,
            print_results=print_results, 
            display_image=display_image, 
            save_image_path=single_image_path,
            save_animation_path=single_animation_path,
            use_bins=use_bins,
            **kwargs
        )

        if distance is None:
            continue
        sequences.append(sequence)
        paths.append(path)
        distances.append(distance)
    
    if output_df:
        # TODO : Implement
        pass
    
    return distances, sequences, paths


def compare_to_optimized(
    domain_id_or_graph, 
    assignment_data,
    fixed_start=False, 
    fixed_end=False, 
    algorithm="annealing", 
    random_seed=None, 
    cumulative_distances=True,
    output_df=False,
    save_csv=False,
    print_results=False,
    display_images=False,
    save_image_path="",
    save_animation_path="",
    print_summary=True,
    use_bins=True,
    **kwargs
):
    """
    Optimize a pick sequence and compare results to the orginal sequence.

    Most useful in the context of `compare_to_optimized_multi`.
    kwargs are passed on to the optimization algorithm.

    """
    if len(assignment_data) < 3:
        print("Fewer than 3 locations in input data; skipping assignment (can't be optimized).")
        return None

    check_algorithm(algorithm)
    G = parse_domain_input(domain_id_or_graph)
    assert nx.is_connected(G), "Graph is not connected; problem may be ill-posed"

    if random_seed:
        np.random.seed(random_seed)
    
    original_distance, original_path = analyze_picks(
        G, 
        assignment_data,
        cumulative_distances,
        None,
        None,
        output_df,
        save_csv, 
        print_results, 
        display_images, 
        get_single_image_path(save_image_path, "_unoptimized"), 
        get_single_image_path(save_animation_path, "_unoptimized", 'gif'),
        use_bins
    )
    od_comparison = original_distance[-1] if cumulative_distances else original_distance

    optimized_distance, optimized_sequence, optimized_path = optimize_picks(
        G,
        assignment_data,
        fixed_start=fixed_start,
        fixed_end=fixed_end,
        algorithm=algorithm,
        cumulative_distances=cumulative_distances,
        output_df=output_df,
        save_csv=save_csv,
        print_results=print_results,
        display_image=display_images,
        save_image_path=get_single_image_path(save_image_path, "_optimized"),
        save_animation_path=get_single_image_path(save_animation_path, "_optimized", 'gif'),
        use_bins=use_bins,
        **kwargs | {'baseline_distance': od_comparison}
    )

    if print_summary:
        print("Original sequence: ", assignment_data)
        print("Optimized sequence: ", optimized_sequence)
        print("Original distance: ", original_distance)
        print("Optimized distance: ", optimized_distance)
        print("Gain: ", original_distance - optimized_distance)

    if output_df:
        # TODO : Implement
        pass

    return {
        "original": {"distance": original_distance, "sequence": assignment_data, "path": original_path},
        "optimized": {"distance": optimized_distance, "sequence": optimized_sequence, "path": optimized_path}
    }


def compare_to_optimized_multi(
    domain_id_or_graph, 
    assignment_data_or_path,
    fixed_start=True, 
    fixed_end=True, 
    algorithm="annealing", 
    random_seed=None, 
    cumulative_distances=False,
    output_df=False,
    save_csv=False,
    print_results=False,
    display_images=False,
    save_image_path="",
    save_animation_path="",
    random_assignments=False,
    assignment_length_range=None,
    num_assignments=None,
    print_summary=True,
    display_comparison_plot=True,
    save_comparison_plot_path="",
    use_bins=True,
    **kwargs
):
    """
    Compare multiple original assignment sequences to optimized sequences.

    By default, displays a bar graph comparing average optimized distance to baseline (original) distances.
    All distances take waypoint <--> bin distance into account.
    
    """
    
    G = parse_domain_input(domain_id_or_graph)
    assert nx.is_connected(G), "Graph is not connected; problem may be ill-posed"
    assignment_data = parse_assignment_input(assignment_data_or_path, random_assignments, assignment_length_range, num_assignments)

    if random_seed:
        np.random.seed(random_seed)
    
    original = {"distances": [], "sequences": [], "paths": []}
    optimized = {"distances": [], "sequences": [], "paths": []}

    for idx, assignment in enumerate(assignment_data):
        comparison_dict = compare_to_optimized(
            G, 
            assignment, 
            fixed_start=fixed_start, 
            fixed_end=fixed_end, 
            algorithm=algorithm,
            cumulative_distances=cumulative_distances,
            output_df=output_df,
            save_csv=save_csv,
            print_results=print_results, 
            display_images=display_images, 
            save_image_path=get_single_image_path(save_image_path, idx),
            save_animation_path=get_single_image_path(save_animation_path, idx, 'gif'),
            print_summary=print_summary,
            use_bins=use_bins,
            **kwargs
        )
        if comparison_dict is None:
            continue

        for key in original:
            original[key].append(comparison_dict["original"][key[:-1]])
            optimized[key].append(comparison_dict["optimized"][key[:-1]])
    
    if display_comparison_plot or save_comparison_plot_path:
        if len(original["distances"]) == 0:
            print("No pick assignments were processed!")
            return None

        mean_unoptimized = np.mean(original["distances"])
        mean_optimized = np.mean(optimized["distances"])

        x = list(range(2))
        y = [mean_unoptimized, mean_optimized]
        plt.title("Mean distance for optimized VS unoptimized paths")
        plt.ylabel("Distance (generic units)")
        plt.bar(x, y)
        add_labels(plt, x, y, upper=False)
        plt.xticks(x, ["Unoptimized", f"Optimized using {algorithm}"])
        if save_comparison_plot_path:
            plt.savefig(save_comparison_plot_path, bbox_inches='tight')
        if display_comparison_plot:
            plt.show()
        
        plt.clf()
        plt.close("all")
    
    if output_df:
        # TODO : Implement
        pass

    return {"original": original, "optimized": optimized}


def analyze_picks_grid(graph, sequence, convert_graph=True, convert_input_to_str=True, node_labels_on=True, **kwargs):
    """
    Alias for `analyze_picks` with `use_bins` set to False (appropriate for grid graphs)

    Parameters same as `analyze_picks` except for the following:

      `convert_graph`: bool: Toggles automatic conversion of graph created with `nx.grid_2d_graph` to the format
        expected by the reslotting algorithm
    
      `convert_input_to_str`: bool: Toggles automatic conversion of input to string

      `node_labels_on`: bool: Included to set node labels on for printing by default


    """
    if convert_graph:
        graph = format_graph(graph)
    if convert_input_to_str:
        sequence = to_str(sequence)
        if 'constraint' in kwargs:
            kwargs['constraint'] = to_str(kwargs['constraint'], recursive=True)
    
    kwargs["node_labels_on"] = node_labels_on
    kwargs["use_bins"] = False
    
    results = analyze_picks(graph, sequence, **kwargs)
    if convert_graph:
        return graph, *results
    return results


def optimize_picks_grid(graph, sequence, convert_graph=True, convert_input_to_str=True, node_labels_on=True, **kwargs):
    """
    Alias for `optimize_picks` with `use_bins` set to False (appropriate for grid graphs)
    Parameters same as for `analyze_picks_grid` immediately above
    """
    if convert_graph:
        graph = format_graph(graph)
    if convert_input_to_str:
        sequence = to_str(sequence)
        if 'constraint' in kwargs:
            kwargs['constraint'] = to_str(kwargs['constraint'], recursive=True)

    kwargs["node_labels_on"] = node_labels_on
    kwargs["use_bins"] = False

    results = optimize_picks(graph, sequence, **kwargs)
    if convert_graph:
        return graph, *results
    return results
