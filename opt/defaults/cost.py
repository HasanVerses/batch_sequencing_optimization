from opt.model.cost import CostFn
from opt.algorithms.kmedoids import within_cluster_shortest_paths, within_cluster_shortest_paths_extranode
from opt.graph import get_distance_matrix
from functools import partial
import numpy as np

MIN_VAL = -64.0 # value used in numerically-stable log transform

def default_cost_weights(num_costs, normalize=True):
    denominator = num_costs if normalize else 1.0
    return [1.0/denominator] * num_costs

def shortest_path_cost(distances_matrix, sequence):
    distance = 0
    for idx in range(len(sequence)-1):
        distance += distances_matrix[sequence[idx]][sequence[idx+1]]

    return distance

def max_cluster_size(cluster_assignments):
    return max([len(cluster) for cluster in cluster_assignments])

def max_cluster_size_cost(cluster_assignments, target_value, scale_fn=None):
    """Compute squared error between the largest cluster assignment and a target value"""
    return compute_minkowski_dist_scalar(max_cluster_size(cluster_assignments), target_value, scale_fn=scale_fn)

def convert_to_batch_size_dist(cluster_assignments, bin_centers):
    """
    This function takes the current cluster assignments and converts them to a probability distribution of batch sizes,
    using some provided `bin_centers`. 
    """

    batch_sizes = [len(assign_ids) for assign_ids in cluster_assignments] # compute the vector of batch sizes

    bin_edges = np.append(bin_centers, bin_centers[-1]+1) # turn bin centers into bin edges (leftmost and rightmost on either 'end' of the vector of bin centers)

    bin_probs, _ = np.histogram(batch_sizes, bins = bin_edges, density = True) # use numpy.histogram function to bin the current vector of batch sizes into the different bins and compute their respective probabilities

    return bin_probs


### @NOTE: Putting things like `log_stable` and `compute_discrete_kullback_leibler` in this cost defaults library is a bit odd -- we should maybe put this into a general utilities library, if one of those exists already?
def log_stable(arr):
    """ numerically stable version of the log transform that adds small amount of epsilon number to zero-valued entries of the array to
    avoid numerical underflow """

    # return np.where(arr, np.log(arr), MIN_VAL)
    return np.log(arr + 1e-32)

def compute_discrete_kullback_leibler(p, q):
    """ 
    Computes the Kullback-Leibler divergence between two discrete probability distributions (vectors of probabilities, e.g.
    Categorical/multinomial distributions) 
    """

    return (log_stable(q) - log_stable(p)).dot(p) # computed as sign-flipped version of the normal KLD expression, typically: -E_{p}[ln(p/q)] = -E_{p}[ln(p) - ln(q)]

def compute_batch_size_divergence(cluster_assignments, reference_batch_size_dist):
    """
    Computes the batch_size divergence between the current distribution of batch sizes (the size of each cluster)
    and the 'reference' distribution that you're trying to diverge against
    """

    # extract the array of bin values and probabilities from the reference distribution
    array_q = np.array(list(reference_batch_size_dist.items()))
    bin_values_q, q_probs = array_q[:,0], array_q[:,1]

    # convert list of current `cluster_assignments` to a probability histogram with same bins as the reference batch size distribution
    p_probs = convert_to_batch_size_dist(cluster_assignments, bin_values_q) # convert current cluster size distribution to batch size distribution

    return compute_discrete_kullback_leibler(p_probs, q_probs)

def compute_batch_dist_vec_minkowski(cluster_assignments, reference_batch_size_dist, p=2.0, scale_fn=None):
    """
    Computes the Minkowski distance between the current histogram of batch sizes (the size of each cluster)
    and the 'reference' histogram of batch sizes that you're comparing it to. Default value of `p` is 2.0, meaning 
    it defaults to the MSE (mean-squared error, Euclidean distance)
    """

    # extract the array of bin values and probabilities from the reference distribution
    array_q = np.array(list(reference_batch_size_dist.items()))
    bin_values_q, q_probs = array_q[:,0], array_q[:,1]

    # convert list of current `cluster_assignments` to a probability histogram with same bins as the reference batch size distribution
    p_probs = convert_to_batch_size_dist(cluster_assignments, bin_values_q) # convert current cluster size distribution to batch size distribution

    return compute_minkowski_dist_vec(p_probs, q_probs, p = p, scale_fn=scale_fn)

def compute_assignment_vec_minkowski(cluster_assignments, reference_vec, p = 2.0, scale_fn=None):
    """
    Computes the Minkowski distance between the current vector of batch sizes (the size of each cluster)
    and the 'reference' assignment size vector that you're comparing it to. Default value of `p` is 2.0, meaning 
    it defaults to the MSE (mean-squared error, Euclidean distance)
    """

    batch_size_array = np.array([len(assign_ids) for assign_ids in cluster_assignments]) # compute the vector of batch sizes
    batch_size_array = np.sort(batch_size_array)

    reference_array = np.sort(np.array(reference_vec))

    if len(batch_size_array) > len(reference_array):
        vec1 = batch_size_array[:(len(reference_array))]
        vec2 = reference_array
        
    else:
        vec1 = batch_size_array
        vec2 = reference_array[:(len(batch_size_array))]

    out = compute_minkowski_dist_vec(vec1, vec2, p = p, scale_fn=scale_fn)

    return out

def compute_clustersize_minkowski(cluster_assignments, desired_size, p=2.0, scale_fn=None):
    """
    Computes the Minkowski distance between each element in the current vector of batch sizes (the size of each cluster)
    and a "reference" desired cluster size `desired_size`. Default value of p is 2.0, meaning it
    defaults to a squared error or MSE cost 
    """
    batch_size_array = np.array([len(assign_ids) for assign_ids in cluster_assignments]) # compute the vector of batch sizes and turn into numpy array
    out = compute_minkowski_dist_vec(batch_size_array, desired_size * np.ones_like(batch_size_array), p=p, scale_fn=scale_fn)
    return out

def compute_minkowski_dist_vec(vec1, vec2, p = 2.0, scale_fn=None):
    """ Computes Minkowski distance between two vectors with order parameter `p` """
    vec_difference = np.absolute(vec1 - vec2)    
    return (scale_fn(vec_difference)**p).sum() if scale_fn else (vec_difference**p).sum()

def compute_minkowski_dist_scalar(val1, val2, p = 2.0, scale_fn=None):
    """ Computes Minkowski distance between two scalars with order parameter `p` """
    scalar_difference = np.absolute(val1 - val2)
    return scale_fn(scalar_difference)**p if scale_fn else scalar_difference**p

class GraphDistanceCost(CostFn):
    def __init__(self, graph, **kwargs):
        graph_distance_matrix = get_distance_matrix(graph)
        distance_cost = partial(shortest_path_cost, graph_distance_matrix)
        super(GraphDistanceCost, self).__init__(cost_fn_handle=distance_cost, **kwargs)

class GraphClusterCost(CostFn):
    """
    Default cost function class for use in graph clustering. The cost function used, `within_cluster_shortest_paths`
    (from the `app.algorithms.kmedoids` module) computes the sum (across clusters) of the sums (within-cluster) of
    shortest-path lengths between nodes within that cluster.
    """
    def __init__(self, graph, **kwargs):
        graph_distance_matrix = get_distance_matrix(graph)
        distance_cost = partial(within_cluster_shortest_paths, graph_distance_matrix)
        super(GraphClusterCost, self).__init__(cost_fn_handle=distance_cost, **kwargs)

class GraphTaskClusterCost(CostFn):
    """
    Like `GraphClusterCost` but considers only the nodes relevant to a specified task (subset of nodes),
    but where the full graph is needed to define shortest path distances
    """
    # NOTE: This could take (graph, task_list) as inputs too. That's probably better, but for now I'm 
    # computing the distance matrix outside this cost definition because it's needed for init anyway.
    # We probably want some nice streamlined way to create valid initializations automatically along
    # with cost definitions.
    def __init__(self, task_distance_matrix, average_flag=True, **kwargs):
        distance_cost = partial(within_cluster_shortest_paths, distances_matrix=task_distance_matrix, average=average_flag)
        super(GraphTaskClusterCost, self).__init__(cost_fn_handle=distance_cost, **kwargs)

class GraphTaskClusterCost_ExtraNode(CostFn):
    """
    Like `GraphTaskClusterCost` but also measures the sum of SPLs, for each node within each cluster,
    to an extra node ID (e.g. a Delivery Zone) id given by `extra_node_id`.
    """
    # NOTE: This could take (graph, task_list) as inputs too. That's probably better, but for now I'm 
    # computing the distance matrix outside this cost definition because it's needed for init anyway.
    # We probably want some nice streamlined way to create valid initializations automatically along
    # with cost definitions.
    def __init__(self, task_distance_matrix, extra_node_id, average_flag=True, **kwargs):

        add_extra_node_to_one = lambda cluster_ids: list(set(cluster_ids + [extra_node_id])) # quick function that adds the extra node_id to a single cluster
        add_extra_id_to_each = lambda all_assignments: list(map(add_extra_node_to_one, all_assignments)) # this acts on a whole list of cluster assignments, adds the extra node id to each cluster-specific assignment in the list of assignments
        distance_cost = partial(within_cluster_shortest_paths, distances_matrix=task_distance_matrix, average=average_flag)
        super(GraphTaskClusterCost_ExtraNode, self).__init__(cost_fn_handle=distance_cost, encoder_fn_handle=add_extra_id_to_each, **kwargs)

class GraphTaskClusterCost_ExtraNode_v2(CostFn):
    """
    Like `GraphTaskClusterCost` but also adds the 2 * the argmin_{node}(SPL) between nodes in a cluster and
    and extra node ID (e.g. a Delivery Zone) id given by `extra_node_id`.
    """
    # NOTE: This could take (graph, task_list) as inputs too. That's probably better, but for now I'm 
    # computing the distance matrix outside this cost definition because it's needed for init anyway.
    # We probably want some nice streamlined way to create valid initializations automatically along
    # with cost definitions.
    def __init__(self, task_distance_matrix, extra_node_id, average_flag=True, **kwargs):

        distance_cost = partial(within_cluster_shortest_paths_extranode, distances_matrix=task_distance_matrix, extra_node=extra_node_id, average=average_flag)
        super(GraphTaskClusterCost_ExtraNode_v2, self).__init__(cost_fn_handle=distance_cost, **kwargs)

class numClusterCost(CostFn):
    """
    Cost function class for use in graph clustering. This cost penalizes the number of clusters 
    for deviating from some "desired" setpoint (`desired_k`) value via a Minkowski penalty
    """
    def __init__(self, desired_k, p=2.0, scale_fn=None, **kwargs):
        cluster_cost = lambda assigns: compute_minkowski_dist_scalar(len(assigns), desired_k, p=p, scale_fn=scale_fn)
        super(numClusterCost, self).__init__(cost_fn_handle=cluster_cost, **kwargs)

class SizeDistCost(CostFn):
    """
    Cost function class for use in graph clustering. This cost penalizes the current cluster
    size distribution for deviating in a information-theoretic sense from some desired distribution
    of cluster sizes (`batch_size_dist`)
    """
    def __init__(self, batch_size_dist, **kwargs):
        size_dist_kl = partial(compute_batch_size_divergence, reference_batch_size_dist = batch_size_dist)
        super(SizeDistCost, self).__init__(cost_fn_handle=size_dist_kl, **kwargs)

class SizeDistCost_PerBin(CostFn):
    """
    Cost function class for use in graph clustering. This cost penalizes the current cluster
    size histogram for deviating in an Minkowski sense (with desired order parameter `p`) from some desired histogram
    of cluster sizes (`batch_size_dist`).
    """
    def __init__(self, batch_size_dist, p=2.0, scale_fn=None, **kwargs):
        size_dist_binwise = partial(compute_batch_dist_vec_minkowski, reference_batch_size_dist = batch_size_dist, p = p, scale_fn=scale_fn)
        super(SizeDistCost_PerBin, self).__init__(cost_fn_handle=size_dist_binwise, **kwargs)

class ClusterSizeMinkowskiCost(CostFn):
    """
    Cost function class for use in graph clustering. This cost penalizes the current vector of
    cluster sizes for deviating in a Minkowski error sense (default p = 2.0, aka MSE or squared Euclidean) from some desired
    cluster size `desired_size`
    """
    def __init__(self, desired_size, p=2.0, **kwargs):
        cluster_size_cost = partial(compute_clustersize_minkowski, desired_size=desired_size, p=p)
        super(ClusterSizeMinkowskiCost, self).__init__(cost_fn_handle=cluster_size_cost, **kwargs)

class AssignmentMSECost(CostFn):
    """
    Cost function class for use in graph clustering. This cost penalizes the current vector of
    cluster sizes for deviating in a least-squared error sense from some desired vector of
    of cluster sizes (`assignment_size_vec`)
    """
    def __init__(self, assignment_size_vec, scale_fn=None, **kwargs):
        assignment_mse = partial(compute_assignment_vec_minkowski, reference_vec = assignment_size_vec, p = 2.0, scale_fn=scale_fn)
        super(AssignmentMSECost, self).__init__(cost_fn_handle=assignment_mse, **kwargs)

class AssignmentManhattanCost(CostFn):
    """
    Cost function class for use in graph clustering. This cost penalizes the current vector of
    cluster szies for deviating in a L1 error (Manhattan distance) sense from some desired vector of
    of cluster sizes (`assignment_size_vec`)
    """
    def __init__(self, assignment_size_vec, scale_fn=None, **kwargs):
        assignment_manhattan = partial(compute_assignment_vec_minkowski, reference_vec = assignment_size_vec, p = 1.0, scale_fn=scale_fn)
        super(AssignmentManhattanCost, self).__init__(cost_fn_handle=assignment_manhattan, **kwargs)

class SpinGlassCost(CostFn):

    def __init__(self, J, antiferro = False):

        def ising_cost(state, J, antiferro = False):
            """
            Linear Hamiltonian that defines an Ising model, with keyword argument
            that turns all interactions negative (makes thing 'antiferromagnetic)
            """
            sign = -1.0 if antiferro else 1.0
            return -0.5 * (state.T @ (sign * J) @ state)
        
        super(SpinGlassCost, self).__init__(cost_fn_handle = ising_cost, cost_args = [J], cost_kwargs = {'antiferro': antiferro})

class MaxClusterSizeCost(CostFn):
    """
    Cost that penalizes assignments based on the MSE between the largest batch's size and a single target value 
    """
    def __init__(self, target_value, scale_fn=None, **kwargs):
        max_size_cost = partial(max_cluster_size_cost, target_value=target_value, scale_fn=scale_fn)
        super(MaxClusterSizeCost, self).__init__(cost_fn_handle=max_size_cost, **kwargs)

class ClusterSizeCost(CostFn):
    """
    Cost that penalizes assignments based on the summed absolute error between batch sizes and a single target batch size 
    """
    def __init__(self, target_value, scale_fn=None, **kwargs):
        max_size_cost = partial(compute_clustersize_minkowski, desired_size=target_value, p=1.0, scale_fn=scale_fn)
        super(ClusterSizeCost, self).__init__(cost_fn_handle=max_size_cost, **kwargs)
