import numpy as np

from opt.tasks import random_task
from opt.io.local import get_domain_graph
from opt.defaults.cost import GraphTaskClusterCost_ExtraNode_v2, ClusterSizeCost, MaxClusterSizeCost
from opt.model.optimizer import Optimizer
from opt.algorithms.kmedoids import initialize_task_assignments
from opt.defaults.modifier import SwappingMergingMix

from opt.visual.visual import plot_graph_with_cluster_ids, plot_overlaid_assignment_histograms, plot_overlaid_assignment_vectors

LD = get_domain_graph("LD")

DZ_waypoint = LD.bins['Delivery zone']['closest_waypoint']

num_tasks = 250
start, end, locations = random_task(LD, num_tasks)

num_clusters = 10

print("Initializing cluster assignments...")
cluster_assigns, task_distance_matrix, from_dist_map, inverse_map, to_idx_map = \
    initialize_task_assignments(LD, [start] + locations + [end], max_k = num_clusters, extra_node = DZ_waypoint)

print("initial", [len(x) for x in cluster_assigns])
distanceCost = GraphTaskClusterCost_ExtraNode_v2(task_distance_matrix, extra_node_id = inverse_map[DZ_waypoint])
print("initial distance cost", distanceCost.eval(cluster_assigns))
sizeCost = ClusterSizeCost(3)
print("initial size cost", sizeCost.eval(cluster_assigns))

opt = Optimizer(costs = [distanceCost, sizeCost],
                cost_weights = [0.99, 0.01],  
                modifier = SwappingMergingMix(task_distance_matrix)
            )

result = opt.optimize(cluster_assigns)
result_mapped = [[to_idx_map[from_dist_map[n]] for n in r] for r in result]

print("result:", result_mapped)
print("optimized distance: ", distanceCost.eval(result))
print("optimized size cost: ", sizeCost.eval(result))
print("optimized number of clusters: ", len(result))
print("optimized cluster sizes: ", [len(x) for x in result])

plot_graph_with_cluster_ids(LD, result_mapped, vary_sizes=True, color_skew_factor=20, save_path='dist_graph_35_p.png')
