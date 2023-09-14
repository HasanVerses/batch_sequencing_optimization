import numpy as np
import networkx as nx
from functools import partial
from copy import deepcopy

from opt.graph import get_closest_waypoint_multi


def assign_ids_step(D, curr_medoids):
    """
    Assignment step in k-medoids algorithm on graphs. All nodes are assigned to the medoid / cluster, to which they have the lowest shortest path length.

    Parameters: 

        `D` [numpy.ndarray]: size (N, N) matrix of shortest-path-lengths between nodes in the graph.

        `curr_medoids` [numpy.ndarray] size (num_clusters, ) vector containing the integer node_ids of the current medoids
    Returns:
       
       `C_idx` [list of lists]: a list of the cluster assignments, where each entry `C_idx[k_i]` is a list of node_ids that have been assigned to the `k_i`-th medoid.

    """

    C_idx = [[] for k_i in range(len(curr_medoids))] # initialize list of assignments

    for node_i in range(D.shape[0]):
        k_id = np.argmin(D[node_i, curr_medoids]) # finds which of the current medoids `node_i` is closest to in terms of minimum of shortest paths to each of the K medoids
        C_idx[k_id].append(node_i)
    
    return C_idx

def update_medoids_step(D, C_idx, curr_medoids):
    """
    Medoid update step in k-medoids algorithm on graphs. New medoids are computed as the node within each cluster, which has the minimal "distance" to the other nodes,
    where distance here is defined as the sum of the shortest path lengths to all other nodes.

    Parameters: 

        `D` [numpy.ndarray]: size (N, N) matrix of shortest-path-lengths between nodes in the graph.

        `C_idx` [list of lists]: a list of the cluster assignments, where each entry `C_idx[k_i]` is a list of node_ids that have been assigned to the `k_i`-th medoid.

        `curr_medoids` [numpy.ndarray size]: (num_clusters, ) vector containing the integer node_ids of the current medoids

    Returns:
       
        `new_medoids` [numpy.ndarray size]: (num_clusters, ) vector containing the integer node_ids of the new, updated medoids

    """

    new_medoids = np.zeros_like(curr_medoids, dtype = int)
    for k_i in range(len(curr_medoids)):
        D_med = D[np.ix_(C_idx[k_i], C_idx[k_i])] # shortest-path distance matrix only amoung those nodes within an assignment
        sum_shortest_paths_per_node = np.zeros(D_med.shape[0]) # initialize an array containing the sum of the shortest paths from each node to all other nodes within the cluster

        if len(sum_shortest_paths_per_node) > 1:
            for c_ii in range(len(sum_shortest_paths_per_node)):
                all_shortest_paths = D_med[c_ii, :]
                paths_no_infs = all_shortest_paths[all_shortest_paths != np.inf]
                sum_shortest_paths_per_node[c_ii] = paths_no_infs.sum()
            new_medoids[k_i] = C_idx[k_i][np.argmin(sum_shortest_paths_per_node)]            
        else:
            new_medoids[k_i] = C_idx[k_i][0]

    return new_medoids
    

def run_k_medoids_graph(D, max_k, num_iter, verbose = False):
    """
    Runs the k-medoid graph clustering / graph partitioning algorithm, given a matrix of shortest-path-lengths between nodes, a fixed number of clusters or medoids to
    cut the graph into, and a number of iterations to run the algorithm.

    Parameters: 

        `D` [numpy.ndarray]: size (N, N) matrix of shortest-path-lengths between nodes in the graph.

        `max_k` [int]: a list of the cluster assignments, where each entry `C_idx[k_i]` is a list of node_ids that have been assigned to the `k_i`-th medoid.

        `num_iter` [int]:  number of iterations to run the algorithm 

        `verbose` [bool]: Flag indicating whether to announce the number of iterations after which the algorithm stopped changing the solution (convergence)

    Returns:
       
        `last_medoids` [numpy.ndarray]: size (num_clusters, ) vector containing the integer node_ids of the final medoids

        `C_idx` [numpy.ndarray]: a list of the final cluster assignments, where each entry `C_idx[k_i]` is a list of node_ids that have been assigned to the `k_i`-th medoid.

    """

    N = D.shape[0]

    init_medoids = np.random.choice(np.arange(N), size = max_k) # randomly assign initial medoids
    C_idx = [[] for k_i in range(max_k)] # assignments
    last_medoids = init_medoids.copy()

    iter_count = 0
    keep_running = True
    while keep_running:

        C_idx = assign_ids_step(D, last_medoids)

        medoid_prune_idx = np.array([len(c_k) > 0 for c_k in C_idx])

        if medoid_prune_idx.sum() < max_k:

            if verbose:
                print('Warning...at least one cluster was not assigned any IDs...pruning clusters\n')
            last_medoids = last_medoids[medoid_prune_idx]
            C_idx_new = []
            for c_k in C_idx:
                if len(c_k) > 0:
                    C_idx_new.append(c_k)
            C_idx = C_idx_new

        next_medoids = update_medoids_step(D, C_idx, last_medoids)

        iter_count += 1

        if (next_medoids == last_medoids).all():
            keep_running = False
            if verbose:
                print(f'Solution stopped changing after {iter_count} iterations, stopping early...\n')
        elif iter_count > num_iter:
            keep_running = False
           
        last_medoids = next_medoids


    # final assignment step, given last curr_medoids update
    C_idx = assign_ids_step(D, last_medoids)

    return last_medoids, C_idx

def initialize_assignments(G, max_k):
    """
    Given a networkX graph `G` and a maximum number of clusters to find `max_k`, initialize
    cluster assignments by randomly choosing some nodes to be the cluster
    "medoids", and assigning all other nodes to those clusters based on the shortest paths to
    the medoids
    """

    N = G.number_of_nodes()
    init_medoids = np.random.choice(np.arange(N), size = max_k) # randomly assign initial medoids

    D = nx.floyd_warshall_numpy(G)
    cluster_assignments = assign_ids_step(D, init_medoids)

    return cluster_assignments

def add_extra_node(full_distance_matrix, reduced_distance_matrix, mapping, extra_node, task_nodes):
    row_to_add = full_distance_matrix[mapping[extra_node],task_nodes].reshape(1,-1) # reshape the row to make it into a (1, num_task_locations) row vector
    col_to_add = full_distance_matrix[task_nodes, mapping[extra_node]].reshape(-1,1) # reshape the column to make it into a (num_task_locations, 1) column vector
    # this is gross, but append the final distance to end of `col_to_add` to get the distance between the `extra_node` and itself
    col_to_add = np.vstack((col_to_add, np.array([0.0]).reshape(-1,1)))

    reduced_distance_matrix = np.vstack( (reduced_distance_matrix, row_to_add) )
    reduced_distance_matrix = np.hstack( (reduced_distance_matrix, col_to_add) )

    return reduced_distance_matrix

def initialize_task_assignments(graph, task_locations, max_k, min_assignment_length=None, use_bins=False, from_df=False, extra_node = None):
    """
    Given a list `task_locations` of nodes to visit on a graph `graph` and a maximum number of clusters to find 
    `max_k`, initialize cluster assignments by randomly choosing some nodes to be the cluster "medoids", and 
    assigning all other nodes to those clusters based on the shortest paths to the medoids.
    """
    D = nx.floyd_warshall_numpy(graph)

    node_ID_to_linear_idx_full = {v: idx for idx, v in enumerate(list(graph))}
    df_idx_to_linear_idx_full = None
    if from_df:
        use_bins=True
        task_locations_df = task_locations
        task_locations = list(task_locations['FromLocation'])        
        df_idx_to_linear_idx_full = {idx: node_ID_to_linear_idx_full[graph.bins[record['FromLocation']]['closest_waypoint']] for idx, record in task_locations_df.iterrows()}
    if use_bins:
        assert hasattr(graph, 'bins'), "Graph lacks `bins` property!"
        bin_to_linear_idx_full = {bin: node_ID_to_linear_idx_full[graph.bins[bin]['closest_waypoint']] for bin in graph.bins}
    
    map_to_use = bin_to_linear_idx_full if use_bins else node_ID_to_linear_idx_full
    task_nodes = [map_to_use[location] for location in task_locations]
    D_tasks = D[task_nodes][:,task_nodes] # Build matrix of only task-relevant nodes

    if extra_node is not None:
        D_tasks = add_extra_node(D, D_tasks, map_to_use, extra_node, task_nodes)
        all_nodes = task_locations + [extra_node]
    else: 
        all_nodes = task_locations
        
    init_medoids = np.random.choice(task_locations, size = max_k, replace=False) # randomly assign initial medoids
    linear_idx_reduced_to_node_ID_or_bin = {idx: v for idx, v in enumerate(all_nodes)}
    node_ID_or_bin_to_linear_idx_reduced = {v: k for k, v in linear_idx_reduced_to_node_ID_or_bin.items()}

    medoid_cols = [node_ID_or_bin_to_linear_idx_reduced[location] for location in init_medoids]   # map node names to column indices
    cluster_assignments = assign_ids_step(D_tasks, medoid_cols)

    if min_assignment_length is not None:
        while any([len(x) < min_assignment_length for x in cluster_assignments]):
            cluster_assignments = random_move(cluster_assignments, drop_empty=False)

    return cluster_assignments, D_tasks, linear_idx_reduced_to_node_ID_or_bin, node_ID_or_bin_to_linear_idx_reduced, map_to_use, df_idx_to_linear_idx_full

def compute_within_cluster_distance(cluster_assignments, distances_matrix, average=False):
    """
    Compute sum of shortest paths between nodes assigned to a given cluster (whose node_ids are stored in some iterable `cluster_assignments`).
    `average` flag tells you whether to normalize the sum of shortest paths by the number of edges in the graph 
    """

    n = len(cluster_assignments) # get size of the cluster

    within_cluster_sum_sp = 0. # this will store the total sum of shortest paths within the cluster whose assigned nodes are in `cluster_assignments`

    for node_i in cluster_assignments:
        for node_j in cluster_assignments:
            within_cluster_sum_sp += distances_matrix[node_i][node_j]

    return (within_cluster_sum_sp / (n*(n-1))) if average else within_cluster_sum_sp

def within_cluster_shortest_paths(cluster_assignments, distances_matrix, average=False):
    """ 
    Computes the total "cluster distance cost" 
    for the kmedoids algorithm. This cost is the sum (or average) (across clusters) of the within-cluster sums of shortest-path-lengths (sum of SPLs)
    An optional `average` flag tells you whether to normalize the sum of shortest paths by the number of edges in each cluster.
    """

    compute_within_cluster_distance_G = partial(compute_within_cluster_distance, distances_matrix=distances_matrix, average=average)
    all_cluster_sum_sp = list(map(compute_within_cluster_distance_G, cluster_assignments))

    return sum(all_cluster_sum_sp)

def compute_within_cluster_distance_extranode(cluster_assignments, distances_matrix, extra_node, average=False):
    """
    Compute sum of shortest paths (sum of SPLs) between nodes assigned to a given cluster (whose node_ids are stored in some iterable `cluster_assignments`).
    An optional `average` flag tells you whether to normalize the sum of shortest paths by the number of edges in each cluster. 
    """

    n = len(cluster_assignments) # get size of the cluster

    within_cluster_sum_sp = 0. # this will store the total sum of shortest paths within the cluster whose assigned nodes are in `cluster_assignments`

    for node_i in cluster_assignments:
        for node_j in cluster_assignments:
            within_cluster_sum_sp += distances_matrix[node_i][node_j]
    
    within_cluster_sum_sp += 2*np.min(distances_matrix[extra_node,:])

    return (within_cluster_sum_sp / (n*(n-1)+1)) if average else within_cluster_sum_sp # add 1 to denominator to count the "extra edge" between one of the cluster nodes and the extra node

def within_cluster_shortest_paths_extranode(cluster_assignments, distances_matrix, extra_node, average=False):
    """ 
    Computes the total "cluster distance cost" 
    for the kmedoids algorithm. This cost is the sum (or average) (across clusters) of the within-cluster sums of shortest-path-lengths (sum of SPLs)
    It also adds to this quantity, the shortest SPL between the node in each cluster to some `extra_node` id.
    An optional `average` flag tells you whether to normalize the sum of shortest paths by the number of edges in the graph 
    """

    compute_within_cluster_distance_G = partial(compute_within_cluster_distance_extranode, distances_matrix=distances_matrix, extra_node=extra_node, average=average)
    all_cluster_sum_sp = list(map(compute_within_cluster_distance_G, cluster_assignments))

    return sum(all_cluster_sum_sp)

def sum_shortest_paths_to_node(node_i, other_nodes, graph_distance_matrix):
    """
    Computes the sum of shortest paths between `node_i` and every other node listed in `other_nodes` 
    """
    return sum([graph_distance_matrix[node_i][node_j] for node_j in other_nodes])


def random_move(cluster_assignments, drop_empty=True):
    """
    A possible modification function to be used in clustering algorithms, where you randomly select two clusters
    ClusterA and ClusterB randomly select one node from Cluster A (Node X from Cluster A),
    and then move its assignment to Cluster B -- so now Node X is assigned to Cluster B instead of to Cluster A.

    Sample two clusters:
    ClusterA, ClusterB

    Sample one node from ClusterA
    ClusterA --> Node X

    Move assignment of randomly sampled node to ClusterB:
    NodeX --> Cluster B
    """

    modded_assignments = deepcopy(cluster_assignments) # @NOTE: use of deepcopy here very important! Make sure you don't change the cluster assignments list in-place

    if len(cluster_assignments) < 2: # need to check for this otherwise the sampling without replacement step below wiill throw an error
        out = modded_assignments # just return the assignments if there are less than 2 clusters -- because this function doesn't even make sense for < 2 clusters
    else:
        cluster1_id, cluster2_id = np.random.choice(len(cluster_assignments), size = 2, replace = False)

        if len(cluster_assignments[cluster1_id]) == 0: # if `cluster1` is empty, remove it
            if drop_empty:
                modded_assignments.pop(cluster1_id)
        else:
            which_node = np.random.choice(cluster_assignments[cluster1_id])
            # move node from first assignment to the other
            modded_assignments[cluster1_id].remove(which_node)
            modded_assignments[cluster2_id].append(which_node) # note that this means you might append `which_node` to an empty list if `cluster2` is empty
        out = modded_assignments

    return out

def random_swap(cluster_assignments, drop_empty=True):
    """
    A possible modification function to be used in clustering algorithms, where you randomly select two clusters
    ClusterA and ClusterB  randomly select one node from each cluster (Node X from Cluster A, Node Y from Cluster B),
    and then swap their assignments -- so now Node Y is assigned to Cluster A, and Node X is assigned to Cluster B.

    Sample one node each from two clusters:
    ClusterA --> Node X, ClusterB --> Node Y

    Swap assignemnts:
    NodeX --> Cluster B, NodeY --> ClusterA
    """

    modded_assignments = deepcopy(cluster_assignments) # @NOTE: use of deepcopy here very important! Make sure you don't change the cluster assignments list in-place

    if len(cluster_assignments) < 2: # need to check for this otherwise the sampling without replacement step below wiill throw an error
        out = modded_assignments # just return the assignments if there are less than 2 clusters -- because this function doesn't even make sense for < 2 clusters
    else:
        cluster1_id, cluster2_id = np.random.choice(len(cluster_assignments), size = 2, replace = False)

        cluster_1_removed = False
        if len(cluster_assignments[cluster1_id]) == 0: # if `cluster1` is empty, remove it
            if drop_empty:
                modded_assignments.pop(cluster1_id)
                cluster_1_removed = True
        else:
            which_node_cluster1 = np.random.choice(cluster_assignments[cluster1_id])
            modded_assignments[cluster1_id].remove(which_node_cluster1)
            modded_assignments[cluster2_id].append(which_node_cluster1)

        if len(cluster_assignments[cluster2_id]) == 0: # if `cluster2` is empty, remove it
            if drop_empty:
                modded_assignments.pop(cluster2_id)
        else:
            which_node_cluster2 = np.random.choice(cluster_assignments[cluster2_id])
            # NOTE: To @Conor - I moved both `remove` and `append` inside the `cluster_1_removed` condition, which keeps this from crashing (but means sometimes the mutation does nothing). Just a fix because I was getting errors when using this earlier.
            if not cluster_1_removed: # if you've removed `cluster1`, don't append any nodes to it -- consequence of this is that `which_node_cluster2` might just disappear from the clustering algorithm
                modded_assignments[cluster2_id].remove(which_node_cluster2)
                modded_assignments[cluster1_id].append(which_node_cluster2)
        out = modded_assignments

    return out

def random_move_shortest_path_minimize(cluster_assignments, graph_distance_matrix, drop_empty=True):
    """
    A possible modification function to be used in clustering algorithms, where you randomly select one cluster
    ClusterA, randomly select one node from Cluster A (Node X from Cluster A),
    and find a new assignment to one of the clusters such that its assignment minimizes the sum of shortest path lengths between
    Node X and all the nodes in the candidate clusters

    Sample a cluster:
    ClusterA

    Sample one node from ClusterA
    ClusterA --> Node X

    Assign Node X to whichever cluster which minimizes the sum of shortest path lengths between Node X and each node in the cluster
    """
    
    modded_assignments = deepcopy(cluster_assignments) # @NOTE: use of deepcopy here very important! Make sure you don't change the cluster assignments list in-place

    if len(cluster_assignments) < 2: # need to check for this otherwise the sampling without replacement step below wiill throw an error
        out = modded_assignments # just return the assignments if there are less than 2 clusters -- because this function doesn't even make sense for < 2 clusters
    else:
        cluster1_id = np.random.choice(len(cluster_assignments))

        if len(cluster_assignments[cluster1_id]) == 0: # if `cluster1` is empty, remove it
            if drop_empty:
                modded_assignments.pop(cluster1_id)
        else:
            which_node_cluster1 = np.random.choice(cluster_assignments[cluster1_id])
            modded_assignments[cluster1_id].remove(which_node_cluster1)

            other_clusters = list( set(range(len(cluster_assignments))) - set([cluster1_id]))

            best_other_cluster = other_clusters[0]
            sum_shortest_paths_best = sum_shortest_paths_to_node(which_node_cluster1, cluster_assignments[best_other_cluster], graph_distance_matrix)

            # loop over all other clusters and identify which cluster is the "best fit" for `which_node_cluster1` based on sum of shortest paths to all other nodes in the cluster
            for other_clust_ii in other_clusters:
                sum_shortest_paths_new = sum_shortest_paths_to_node(which_node_cluster1, cluster_assignments[other_clust_ii], graph_distance_matrix)
                if sum_shortest_paths_new < sum_shortest_paths_best:
                    best_other_cluster = other_clust_ii
                    sum_shortest_paths_best = sum_shortest_paths_new
            
            modded_assignments[best_other_cluster].append(which_node_cluster1)
    
        out = modded_assignments

    return out

def random_clustersplit(cluster_assignments, drop_empty=True):
    """
    A modification function that randomly chooses a cluster, selects a random half of its nodes (approximately) and splits it into two clusters
    """

    modded_assignments = deepcopy(cluster_assignments) # @NOTE: use of deepcopy here very important! Make sure you don't change the cluster assignments list in-place

    cluster_to_split = np.random.choice(len(cluster_assignments))

    if len(cluster_assignments[cluster_to_split]) == 0: # if `cluster_to_split` is empty, remove it
        if drop_empty:
            modded_assignments.pop(cluster_to_split)
    elif len(cluster_assignments[cluster_to_split]) >= 2: # this ensures that if size of cluster_assignments[cluster_to_split] == 1, then we just return the original assignments
        new_cluster_size = len(cluster_assignments[cluster_to_split]) // 2 # approximately half of current cluster's nodes become a new cluster
        new_cluster_nodes = np.random.choice(cluster_assignments[cluster_to_split], size = (new_cluster_size,), replace=False)
        for new_node_i in new_cluster_nodes:
            modded_assignments[cluster_to_split].remove(new_node_i)

        modded_assignments.append(list(new_cluster_nodes))

    out = modded_assignments

    return out

def random_2clustermerge(cluster_assignments):
    """
    A modification function that randomly chooses two clusters and merges them into one cluster
    """
    modded_assignments = deepcopy(cluster_assignments) # @NOTE: use of deepcopy here very important! Make sure you don't change the cluster assignments list in-place

    if len(modded_assignments) > 1:
        
        clusters_to_split = np.random.choice(len(modded_assignments), size = (2,), replace = False)

        modded_assignments[clusters_to_split[0]] += modded_assignments[clusters_to_split[1]]
        modded_assignments.pop(clusters_to_split[1]) # remove the second cluster (all its nodes have now merged with the first cluster)

    out = modded_assignments

    return out

def add_node_to_cluster(state):
    """ Modifier function that randomly adds a node from the "limbo pool" (pool of currently un-assigned nodes) to one of the clusters """

    modded_state = deepcopy(state) # @NOTE: use of deepcopy here very important! Make sure you don't change the cluster assignments list in-place

    cluster_assignments, pool = modded_state # unpack the `modded_state` into the current set of cluster assignments and the "limbo pool" of currently-unassigned nodes

    if len(pool) > 0: # only choose a node from the limbo pool if it has anything in it
        node_to_add = np.random.choice(pool)

        cluster_to_join = np.random.choice(len(cluster_assignments))

        cluster_assignments[cluster_to_join].append(node_to_add)
        pool.remove(node_to_add)

    return modded_state

def remove_node_from_cluster(state):
    """ Modifier function that randomly removes a node from one of the clusters and adds it to the "limbo pool" (pool of currently un-assigned nodes) """

    modded_state = deepcopy(state) # @NOTE: use of deepcopy here very important! Make sure you don't change the cluster assignments list in-place

    cluster_assignments, pool = modded_state # unpack the `modded_state` into the current set of cluster assignments and the "limbo pool" of currently-unassigned nodes

    cluster_to_strip = np.random.choice(len(cluster_assignments))

    if len(cluster_assignments[cluster_to_strip]) > 0:
        node_to_remove = np.random.choice(cluster_assignments[cluster_to_strip])

        pool.append(node_to_remove)
        cluster_assignments[cluster_to_strip].remove(node_to_remove)

        if len(cluster_assignments[cluster_to_strip]) == 0: # if the cluster is empty after moving its single node to the pool, then remove the cluster entirely
            cluster_assignments.pop(cluster_to_strip)

    return modded_state

def select_closest_clusters(D, curr_medoids):
    closest_nodes = [np.argmin(D[node_i, curr_medoids]) for node_i in curr_medoids]
    min_distances = [D[node_i, node_j] for node_i, node_j in enumerate(closest_nodes)]

    closest_1 = np.argmin(min_distances)
    return closest_1, closest_nodes[closest_1]

def equalize(cluster1, cluster2):
    diff = len(cluster1) - len(cluster2)
    to_shift = diff//2
    if diff > 0:
        chunk = cluster1[-to_shift:]
        cluster1, cluster2 = cluster1[:-to_shift], cluster2 + chunk
    elif diff < 0:
        chunk = cluster2[to_shift:]
        cluster1, cluster2 = cluster2[:to_shift], cluster1 + chunk

    return cluster1, cluster2    

def shift_mass(cluster1, cluster2):
    pass

def shift_clusters(D, cluster_assignments, ref_vec):
    
    modded_assignments = deepcopy(cluster_assignments) # @NOTE: use of deepcopy here very important! Make sure you don't change the cluster assignments list in-place

    len_vec = [len(x) for x in cluster_assignments]
    current_var = np.var(len_vec)
    ref_var = np.var(ref_vec)

    medoids = update_medoids_step(D, cluster_assignments, [c[0] for c in cluster_assignments])
    cluster1, cluster2 = select_closest_clusters(D, medoids)
    
    if current_var > ref_var:
        modded_assignments[cluster1], modded_assignments[cluster2] = equalize(cluster1, cluster2)
    elif ref_var > current_var:
        modded_assignments[cluster1], modded_assignments[cluster2] = shift_mass(cluster1, cluster2)





