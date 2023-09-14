import numpy as np
from scipy.stats import expon

def max_k_equals(cluster_assignments, max_k):
    """ 
    Boolean-output function (e.g., could be passed in as `constraint_fn_handle` to an instance of `ConstraintFn`) that
    checks whether the number of currently-found clusters (accessed by `len(cluster_assignments)`) is equal to a fixed number
    `max_k`
    """
    return True if len(cluster_assignments) == max_k else False

def max_k_between(cluster_assignments, lower_upper):
    """ 
    Boolean-output function (e.g., could be passed in as `constraint_fn_handle` to an instance of `ConstraintFn`) that
    checks whether the number of currently-found clusters (accessed by `len(cluster_assignments)`) within some bounds
    given by entries `[0]` and `[1]` of `lower_upper`
    """
    return True if (len(cluster_assignments) >= lower_upper[0]) and (len(cluster_assignments) <= lower_upper[1]) else False

def min_cluster_size(cluster_assignments, min_size):
    """ 
    Boolean-output function (e.g., could be passed in as `constraint_fn_handle` to an instance of `ConstraintFn`) that
    checks whether the smallest cluster (accessed by finding the minimum of the vector of sizes of each entry of `cluster_assignments` 
    is greater than or equal to `min_size`, the minimum allowable size -- if this were being used in a constrained optimization context.
    """

    return True if min([len(cluster_ids) for cluster_ids in cluster_assignments]) >= min_size else False

def exponential_size_dist(min_batch_size, max_batch_size, loc = 0.0, scale = 1.0):
    """ 
    Computes an exponential assignment (batch) size distribution using a min and max batch size
    """

    bin_centers = np.array(list(range(min_batch_size, max_batch_size+1)))

    probabilities = expon.pdf(bin_centers, loc, scale)
    probabilities /= probabilities.sum()

    # old version that doesn't require scipy
    # lambda_var = 1.0 / scale
    # probabilities = lambda_var * np.exp(-lambda_var * bin_centers)
    # probabilities /= probabilities.sum()

    return dict(zip(bin_centers, probabilities))

