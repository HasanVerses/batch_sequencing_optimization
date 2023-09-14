import numpy as np

from opt.tasks import random_task
from opt.io.local import get_domain_graph
from opt.defaults.cost import GraphTaskClusterCost, numClusterCost, SizeDistCost_PerBin, SizeDistCost, AssignmentManhattanCost, AssignmentMSECost
from opt.model.optimizer import Optimizer
from opt.algorithms.kmedoids import initialize_task_assignments
from opt.model.constraint import ConstraintFn 
from opt.defaults.modifier import clusterSwap, ClusterSwapSmartTask, ClusterSwap, ClusterSplitMerge
from opt.defaults.constraint import max_k_between, min_cluster_size, exponential_size_dist

from opt.visual.visual import plot_graph_with_cluster_ids, plot_overlaid_assignment_histograms, plot_overlaid_assignment_vectors



LD = get_domain_graph("LD")

num_tasks = 250
start, end, locations = random_task(LD, num_tasks)

num_clusters, min_size, max_size = 10, 3, LD.number_of_nodes()

lower_upper = [num_clusters - 2, num_clusters + 2] # allowable bounds for number of clusters

print("Initializing cluster assignments...")
cluster_assigns, task_distance_matrix, from_dist_map, inverse_map, to_idx_map = \
    initialize_task_assignments(LD, [start] + locations + [end], max_k = num_clusters) 

print("initial", [len(x) for x in cluster_assigns])
distanceCost = GraphTaskClusterCost(task_distance_matrix)

clusterCost = numClusterCost(num_clusters)

# Hard constraints on number of clusters (within allowable range) and minimum cluster size (3)
nClustConstraint = ConstraintFn(max_k_between, [lower_upper], hard = True)
minClustSize = ConstraintFn(min_cluster_size, [min_size], hard = True)

# create fa fake `batch_size_dist` that would come from e.g. NRI
batch_size_dist = exponential_size_dist(min_batch_size=min_size, max_batch_size=max_size, loc = min_size, scale = 10.0)
perBin_HistCost = SizeDistCost_PerBin(batch_size_dist, p=2.0)
#klCost = SizeDistCost(batch_size_dist)

# Generate a random vector of `num_clusters` assignment counts that add up to `num_tasks`
ref_vec = np.random.dirichlet(np.ones(num_clusters))*num_tasks
ref_vec = np.clip(np.array([int(np.round(x)) for x in ref_vec]), min_size, 10000)
while sum(ref_vec) > num_tasks:
    idx = np.random.choice(len(ref_vec))
    ref_vec[idx] -= 1
    ref_vec = np.clip(np.array([int(np.round(x)) for x in ref_vec]), min_size, 10000)

print("ref vec", ref_vec)

manhattanCost = AssignmentManhattanCost(assignment_size_vec = ref_vec)
mseCost = AssignmentMSECost(assignment_size_vec = ref_vec)

num_inner_iters = num_tasks*100
max_successes = num_tasks*10
num_iters = num_tasks*2

opt = Optimizer(costs = [distanceCost, clusterCost, mseCost], #perBin_HistCost], #mseCost], #manhattanCost], #perBin_HistCost], #klCost],
                cost_weights = [1.0, 0.2, 5.0],  
                constraints = [minClustSize, nClustConstraint],
                # modifier = clusterSwap,
                #modifier = ClusterSwap(drop_empty=False),
                # num_iters=200,
                # inner_loop_iters=100,
                # modifier = ClusterSwapSmartTask(task_distance_matrix),
                modifier = ClusterSplitMerge(split_prob = 0.5, merge_prob = 0.5),
                # num_iters = num_iters,
                # inner_loop_iters = num_inner_iters,
                # max_successes=max_successes,
                energy_plot_path='dist_energies_35_p.png'
            )

result = opt.optimize(cluster_assigns)
result_mapped = [[to_idx_map[from_dist_map[n]] for n in r] for r in result]

print("result:", result_mapped)
print("optimized distance: ", distanceCost.eval(result))
print("optimized number of clusters: ", len(result))

assign_vec = [len(x) for x in result]
print("number of assignments in result", sum(ref_vec))

plot_graph_with_cluster_ids(LD, result_mapped, vary_sizes=True, color_skew_factor=20, save_path='dist_graph_35_p.png')

#plot_overlaid_assignment_histograms(batch_size_dist, result, bin_axis_offset = 0.2, bar_width = 0.4)
plot_overlaid_assignment_vectors(ref_vec, assign_vec, save_path='dist_sizes_35_p.png', sort=True)
