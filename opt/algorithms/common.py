import numpy as np

from itertools import permutations
from tqdm import tqdm

from opt.graph import get_node_list, get_distance_multi, get_shortest_path_multi, get_closest_waypoint_multi
from opt.utils import conditional_int



def init_cost_fns(constraint_fn, cost_fn, constraint_kwargs, cost_kwargs):
    constraint_fn = constraint_fn or (lambda state, **kwargs: [True])
    cost_fn = cost_fn or (lambda state, **kwargs: [0.])
    constraint_kwargs = constraint_kwargs or dict()
    cost_kwargs = cost_kwargs or dict()

    return constraint_fn, cost_fn, constraint_kwargs, cost_kwargs


def parse_constraint(constraint_list):
    """
    Given a list of priority sets of nodes, return a constraint function that takes the list as 
    input and implements the constraint. 
    Input format: [{priority_1_nodes}, {priority_2_nodes}, ...]
    """

    if len(constraint_list) == 0:
        print("No constraints specified!")
        return None, None
    
    def order_constraint(state, node_sets):
        state = list(state)
        max_idx = 0
        for node_set in node_sets:
            max_idx += len(node_set)
            if any([state.index(n) >= max_idx for n in node_set]):
                return [False]
        return [True]
    
    return order_constraint, {"node_sets": constraint_list}


def compute_cost(constraint_fn, state, fn_kwargs, penalty, hard=True):
    if hard: 
        return sum([not c for c in constraint_fn(state, **fn_kwargs)]) * penalty
    else:
        return sum(constraint_fn(state, **fn_kwargs)) * penalty


def energy(
    graph, 
    state, 
    start, 
    end,
    decoder=None,
    decoder_kwargs=None,
    constraint_fn=None,
    constraint_penalty=100,
    constraint_fn_kwargs=None,
    cost_fn=None,
    cost_penalty=0.2,
    cost_fn_kwargs=None
):
    """Energy function that incorporates generic constraints"""

    constraint_fn, cost_fn, constraint_fn_kwargs, cost_fn_kwargs = init_cost_fns(
        constraint_fn, cost_fn, constraint_fn_kwargs, cost_fn_kwargs)

    node_list = get_node_list(state, start, end)

    if decoder:
        decoder_kwargs = decoder_kwargs or dict()
        node_list = decoder(node_list, **decoder_kwargs)
    
    node_list = convert_duplicate_node_multi(node_list)
    distance = get_distance_multi(graph, node_list)

    return distance + compute_cost(constraint_fn, state, constraint_fn_kwargs, constraint_penalty) + \
        compute_cost(cost_fn, state, cost_fn_kwargs, cost_penalty, hard=False)
    

def concatenation_fn(start, end):
    """Return a function to combine possibly empty start and end values with node identifiers"""
    if start is None and end is None:
        return lambda x, start, end: x
    elif start is not None and end is None:
        return lambda x, start, end: sum([(start,), x],())
    elif start is None and end is not None:
        return lambda x, start, end: sum([x, (end,)],())
    elif start is not None and end is not None:
        return lambda x, start, end: sum([(start,), x, (end,)],())  


def locally_optimal_paths(
    graph, 
    locations, 
    start, 
    end, 
    use_cached_paths=True, 
    decoder=None,
    decoder_kwargs=None,
    constraint_fn=None,
    cost_fn=None,
    constraint_fn_kwargs=None,
    cost_fn_kwargs=None
):

    constraint_fn, cost_fn, constraint_fn_kwargs, cost_fn_kwargs = init_cost_fns(
        constraint_fn, cost_fn, constraint_fn_kwargs, cost_fn_kwargs)
    
    f = concatenation_fn(start, end)    
    perms = list(permutations(locations))
    g = (lambda perm: tuple(
        convert_duplicate_node_multi(
            decoder(perm, **decoder_kwargs)
            )
        )) if decoder is not None else lambda perm: perm

    paths_and_distances = [
        (get_shortest_path_multi(
            graph,
            f(g(perm), start, end),
            False,
            use_cached_paths,
            False
        ), 
        get_distance_multi(
            graph,
            f(g(perm), start, end), 
            False
        ) + sum(cost_fn(f(g(perm), start, end), **cost_fn_kwargs)),
        perm) for perm in tqdm(perms) if all(constraint_fn(perm, **constraint_fn_kwargs))
    ]

    
    return [list(x) for x in zip(*paths_and_distances)]


def valid_initial_state(sequence, constraint):
    """Define a valid (constraint-satisfying) initial state for priority constraints"""
    initial_state = []
    for node_set in constraint:
        initial_state += list(node_set)
    
    return initial_state + [node for node in sequence if node not in initial_state]


def convert_duplicate_node(node, delimiter='*_*'):
    return conditional_int(str(node).split(delimiter)[0])


def convert_duplicate_node_multi(node_list, delimiter='*_*'):
    return [convert_duplicate_node(node, delimiter) for node in node_list]


def create_duplicate_node(node, idx, delimiter='*_*'):
    return f'{node}{delimiter}{idx}'


def create_duplicate_node_multi(node_list, mode="source", delimiter='*_*'):
    assert mode in ["source", "destination"], "`mode` paramter should indicate `source` or `destintation`"
    return_nodes = []
    appeared = []
    if mode == "destination":
        node_list = list(reversed(node_list))
    for node in node_list:
        return_nodes.append(node if node not in appeared else create_duplicate_node(node, appeared.count(node)))
        appeared.append(node)
    if mode == "destination":
        return_nodes = list(reversed(return_nodes))

    return return_nodes


def constraint_waypoints(graph, constraint):
    """Get closest waypoints to bins by priority level in constrained problem"""    
    unique_waypoints = [set(create_duplicate_node_multi(
        get_closest_waypoint_multi(graph, priority_level)
    )) for priority_level in constraint] 

    return unique_waypoints


def null_result(message=None):
    if message:
        print(message)
    return None, None, np.nan


def parse_results(state, start, end, graph, decoder, decoder_kwargs):
    """Derive full set of results from the sequence returned by an approximate TSP solver"""
    raw_answer = get_node_list(state, start, end)
    if decoder is not None:
        decoded_raw_answer = decoder(raw_answer, **decoder_kwargs)
        decoded_raw_answer = convert_duplicate_node_multi(decoded_raw_answer)
        answer = get_shortest_path_multi(graph, decoded_raw_answer)
    else:
        answer = get_shortest_path_multi(graph, convert_duplicate_node_multi(raw_answer))

    return raw_answer, answer, get_distance_multi(graph, answer)


def null_constraint(state, **kwargs):
    return [True]


def null_cost(state, **kwargs):
    return [0.]