import numpy as np

from opt.tasks import random_task
from opt.io.local import get_domain_graph
from opt.defaults.cost import GraphTaskClusterCost_ExtraNode, numClusterCost
from opt.model.optimizer import Optimizer
from opt.algorithms.kmedoids import initialize_task_assignments
from opt.model.constraint import ConstraintFn 
from opt.defaults.modifier import SwappingMergingMix
from opt.defaults.constraint import max_k_between, min_cluster_size

from opt.visual.visual import plot_graph_with_cluster_ids, plot_overlaid_assignment_histograms, plot_overlaid_assignment_vectors

LD = get_domain_graph("LD")

DZ_waypoint = LD.bins['Delivery zone']['closest_waypoint']

num_tasks = 250
start, end, locations = random_task(LD, num_tasks)

num_clusters, min_size, max_size = 10, 3, LD.number_of_nodes()

lower_upper = [num_clusters - 2, num_clusters + 2] # allowable bounds for number of clusters

print("Initializing cluster assignments...")
cluster_assigns, task_distance_matrix, from_dist_map, inverse_map, to_idx_map = \
    initialize_task_assignments(LD, [start] + locations + [end], max_k = num_clusters, extra_node = DZ_waypoint)

print("initial", [len(x) for x in cluster_assigns])
distanceCost = GraphTaskClusterCost_ExtraNode(task_distance_matrix, extra_node_id = inverse_map[DZ_waypoint])

minkowski_order = 2.0
clusterCost = numClusterCost(num_clusters, p = minkowski_order)

# Hard constraints on number of clusters (within allowable range) and minimum cluster size (3)
# nClustConstraint = ConstraintFn(max_k_between, [lower_upper], hard = True)
# minClustSize = ConstraintFn(min_cluster_size, [min_size], hard = True)

opt = Optimizer(costs = [distanceCost, clusterCost],
                cost_weights = [1.0, 0.2],  
                # constraints = [minClustSize, nClustConstraint],
                modifier = SwappingMergingMix(task_distance_matrix, drop_empty=True)
            )

result = opt.optimize(cluster_assigns)
result_mapped = [[to_idx_map[from_dist_map[n]] for n in r] for r in result]

print("result:", result_mapped)
print("optimized distance: ", distanceCost.eval(result))
print("optimized number of clusters: ", len(result))

assign_vec = [len(x) for x in result]

plot_graph_with_cluster_ids(LD, result_mapped, vary_sizes=True, color_skew_factor=20, save_path='dist_graph_35_p.png')

