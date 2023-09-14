import numpy as np
from functools import partial
from opt.algorithms.genetic import exchange, reverse
from opt.algorithms.kmedoids import (
    random_move, 
    random_swap, 
    random_move_shortest_path_minimize, 
    random_clustersplit, 
    random_2clustermerge
)
from opt.graph import get_distance_matrix

from opt.model.modifier import ModifierFn


class ClusterSwapSmartGraph(ModifierFn):
    def __init__(self, graph):
        graph_distance_matrix = get_distance_matrix(graph)
        random_smart_move = partial(random_move_shortest_path_minimize, graph_distance_matrix=graph_distance_matrix)
        super(ClusterSwapSmartGraph, self).__init__(mod_fn_handles=[random_smart_move], mod_probs=[1.0])

class ClusterSwapSmartTask(ModifierFn):
    #NOTE: Same as for GraphTaskClusterCost: could init with (graph, task_list)
    # Maybe we want some function in a `graph optimization utils` package that generates 
    # (ClusterSwapSmartTask, GraphTaskClusterCost, initial_state) given (graph, task_list) 
    def __init__(self, distance_matrix, drop_empty=True):
        random_smart_move = partial(
            random_move_shortest_path_minimize, 
            graph_distance_matrix=distance_matrix,
            drop_empty=drop_empty
        )
        super(ClusterSwapSmartTask, self).__init__(mod_fn_handles=[random_smart_move], mod_probs=[1.0])

geneticMutations = ModifierFn([exchange, reverse], [0.5, 0.5])

clusterSwap = ModifierFn([random_move, random_swap], [0.9, 0.1])

#NOTE: I added this mainly so I could test a 'pure' distribution-matching objective without the distance
# or other costs, and I wanted to be able to turn off the cluster-dropping thing to be sure we stayed
# at the right num-k. Not opposed to dropping this or reformulating later
class ClusterSwap(ModifierFn):
    """Default cluster swap class that allows some customization of hyperparams"""
    def __init__(self, drop_empty, mod_probs=[0.9, 0.1]):
        move = partial(random_move, drop_empty=drop_empty)
        swap = partial(random_swap, drop_empty=drop_empty)
        super(ClusterSwap, self).__init__(mod_fn_handles=[move, swap], mod_probs=mod_probs)

class ClusterSplitMerge(ModifierFn):
    """ default ModifierFn sub-class that splits or merges clusters randomly """

    def __init__(self, split_prob = 0.5, merge_prob = 0.5):

        if (split_prob + merge_prob) != 1.0:
            raise ValueError("split_prob and merge_prob must sum to 1.0!")
        
        super(ClusterSplitMerge, self).__init__(mod_fn_handles = [random_clustersplit, random_2clustermerge], mod_probs = [split_prob, merge_prob])

class ClusterShift(ModifierFn):
    """Measure the variance of the current and reference distribution of batch sizes, and shift mass """
    # TODO: Finish
    def __init__(self, variance_):
        pass

class ClusterSwapSmartTaskStochastic(ModifierFn):
    #NOTE: Same as for GraphTaskClusterCost: could init with (graph, task_list)
    # Maybe we want some function in a `graph optimization utils` package that generates 
    # (ClusterSwapSmartTask, GraphTaskClusterCost, initial_state) given (graph, task_list) 
    def __init__(self, distance_matrix, mod_probs=[0.36, 0.04, 0.6], drop_empty=False): #0.09, 0.01, 0.9
        random_smart_move = partial(
            random_move_shortest_path_minimize, 
            graph_distance_matrix=distance_matrix,
            drop_empty=drop_empty
        )
        rm = partial(random_move, drop_empty=drop_empty)
        rs = partial(random_swap, drop_empty=drop_empty)
        super(ClusterSwapSmartTaskStochastic, self).__init__(
            mod_fn_handles=[rm, rs, random_smart_move], 
            mod_probs=mod_probs,
        )

class SwappingMergingMix(ModifierFn):
    """
    Mix of random moves, swaps, smart moves and cluster split/merge moves
    """
    def __init__(self, distance_matrix, mod_probs=[0.20, 0.04, 0.6, 0.06, 0.10], drop_empty=True): #0.09, 0.01, 0.9
        random_smart_move = partial(
            random_move_shortest_path_minimize, 
            graph_distance_matrix=distance_matrix,
            drop_empty=drop_empty
        )
        rm = partial(random_move, drop_empty=drop_empty)
        rs = partial(random_swap, drop_empty=drop_empty)

        r_csplit = random_clustersplit
        r_cmerge = random_2clustermerge

        super(SwappingMergingMix, self).__init__(
            mod_fn_handles=[rm, rs, random_smart_move, r_csplit, r_cmerge], 
            mod_probs=mod_probs,
        )

