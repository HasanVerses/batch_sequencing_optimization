from opt.graph import get_node_list
from opt.algorithms.common import locally_optimal_paths, init_cost_fns



def naive_tsp(
    graph, 
    locations, 
    start=None, 
    end=None, 
    use_cached_paths=True, 
    constraint_fn=None, 
    cost_fn=None,
    decoder=None,
    decoder_kwargs=None,
    constraint_fn_kwargs=None,
    cost_fn_kwargs=None
):
    """
    Solve the TSP using the naive / brute-force method.
    By default, node_list[0] and node_list[-1] used as start and end points, respectively
    """
    constraint_fn, cost_fn, constraint_fn_kwargs, cost_fn_kwargs = init_cost_fns(
        constraint_fn, cost_fn, constraint_fn_kwargs, cost_fn_kwargs)

    if decoder is not None and decoder_kwargs is None:
        decoder_kwargs = dict()

    node_list = get_node_list(locations, start, end)

    if start is not None:
        node_list = node_list[1:]
    
    if end is not None:
        node_list = node_list[:-1]

    paths, distances, perms = locally_optimal_paths(
        graph, 
        node_list,
        start,
        end, 
        use_cached_paths,
        decoder=decoder,
        decoder_kwargs=decoder_kwargs,
        constraint_fn=constraint_fn,
        cost_fn=cost_fn,
        constraint_fn_kwargs=constraint_fn_kwargs,
        cost_fn_kwargs=cost_fn_kwargs
    )
    min_distance = min(distances)
    best_idx = distances.index(min_distance)
    the_path = paths[best_idx]
    sequence = get_node_list(perms[best_idx], start, end)

    return sequence, the_path, min_distance
