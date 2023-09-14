import numpy as np
import networkx as nx
from multiprocessing import Pool

from opt.algorithms.annealing import anneal, deterministic_anneal, anneal_thermodynamic
from opt.algorithms.genetic import genetic, genetic_crossover
from opt.algorithms.naive import naive_tsp
from opt.algorithms.common import parse_constraint, null_constraint, null_cost



MAX_PARALLEL_PROCESSES = 512


def solve_tsp(
    graph, 
    locations, 
    start=None, 
    end=None, 
    constraint=None, 
    algorithm="annealing",
    n=1, 
    return_all_results=False,
    random_seed=None, 
    **kwargs
):
    """
    Main function for solving (constrained) TSP with a specified algorithm.
    Not intended for use on bin location data directly.

    Parameters:

      `graph`: networkX graph: represents the space on which locations are defined - must be a connected graph

      `locations`: List (len >= 3): unique node identifiers (e.g. int or str) specifying the locations 
        on the graph that must be visited. Includes first and last nodes on the path unless `start` / `end` 
        parameters are supplied.

      `start`: int: Node identifier for fixed start node in a tour. By efault, start node is initialized 
        as locations[0] and allowed to vary freely

      `end`: int: Node identifier for fixed end node in a tour. By default, end node 
        is initialized as locations[-1] and allowed to vary freely

      `constraint`: List[set]: List of sets of node identifiers to treat with priority.
        Format: [{priority_1_nodes}, {priority_2_nodes}, ...]

        The algorithm will enforce the constraint that all nodes in {priority_1_nodes} are visited before
        the rest, that all nodes in {priority_2_nodes} are visited before the remaining ones, etc.

      `algorithm`: str: specifies which algorithm to use:
        "annealing": Simulated annealing (see `anneal` definition for kwargs)
        "threshold": Deterministic simulated annealing (see `algorithms.annealing.deterministic_anneal`)
        "thermodynamic": Simulated annealing with thermodynamic annealing schedule 
          (see `algorithms.annealing.anneal_thermodynamic`)
        "genetic": Genetic algorithm (see `genetic` definition for kwargs)
        "genetic x": Genetic algorithm with crossover (see `genetic_crossover`)
          definition for kwargs
        "naive": Naive / brute-force solution (see `naive_tsp` definition for 
          kwargs)

      `n`: int: Number of runs of the algorithm to perform in parallel. By default, best (minimum-distance)
        result will be returned.

      `return_all_results`: bool: Toggles whether to return all results from multiple parallel runs. If True,
        the return values will be lists, where index 0 corresponds to the first run, etc.

      `random_seed`: int specifying random seed to use for numpy.

    Returns:

        (`sequence`, `path`, `distance`), where:

        `sequence` = Sequence of node identifiers representing the path through 
            the graph found by the algorithm (only includes nodes in the input 
            list, not intermediate ones visited along the path)

        `path` = Sequence of node identifiers representing the path through the 
            graph found by the algorithm (including all intermediate nodes 
            traversed)
        
        `distance` = The length of the path (summed edge weights between nodes)
    """

    algorithms = {
        "annealing": anneal,
        "threshold": deterministic_anneal,
        "thermodynamic": anneal_thermodynamic,
        "genetic": genetic,
        "genetic x": genetic_crossover,
        "naive": naive_tsp
    }
    assert algorithm in algorithms, "Please use `annealing`, `threshold`, `genetic`, `genetic x` or `naive`"
    
    assert nx.is_connected(graph), "Graph is not connected; problem may be ill-posed"

    assert n > 0 < MAX_PARALLEL_PROCESSES, "`n` (number of processes) must be between 1 and MAX_PARALLEL_PROCESSES"

    if constraint:
        constraint_fn, constraint_fn_kwargs = parse_constraint(constraint)
        kwargs |= {"constraint_fn": constraint_fn, "constraint_fn_kwargs": constraint_fn_kwargs}

    if "constraint_fn" not in kwargs:
        kwargs |= {"constraint_fn": null_constraint}

    if "cost_fn" not in kwargs:
        kwargs |= {"cost_fn": null_cost}

    if random_seed:
        np.random.seed(random_seed)
    
    algo = algorithms[algorithm]
    if n == 1:
        return algo(graph, locations, start, end, **kwargs)
    else:
        pool = Pool()
        _results = [pool.apply_async(algo, (graph, locations, start, end,), kwargs) for _ in range(n)]
        pool.close()
        pool.join()
        results = np.array([r.get() for r in _results], dtype='object').T

        if return_all_results:
            return [list(x) for x in results]
        else:
            return list(results[:,np.argmin(results[2])])
