from opt.graph import random_graph
from opt.tasks import random_task
from opt.defaults.cost import GraphTaskClusterCost, numClusterCost
from opt.model.optimizer import Optimizer
from opt.algorithms.kmedoids import initialize_task_assignments
from opt.model.constraint import ConstraintFn 
from opt.defaults.modifier import ClusterSwapSmartTask
from opt.defaults.constraint import max_k_between
from opt.visual.visual import plot_graph_with_cluster_ids



print("Defining graph...")
G = random_graph(1200) # create randomly locally-connected rectangular spatial graph
print("Defining optimizer...")
start, end, locations = random_task(G, 250)

num_clusters = 35  # start with a desired number of graph clusters
lower_upper = [num_clusters - 2, num_clusters + 2] # allowable bounds for number of clusters

print("Initializing cluster assignments...")
cluster_assigns, task_distance_matrix, node_map, inverse_map, _ = \
    initialize_task_assignments(G, [start] + locations + [end], max_k = num_clusters) 

distanceCost = GraphTaskClusterCost(task_distance_matrix)

clusterCost = numClusterCost(num_clusters)

nclustConstraint = ConstraintFn(max_k_between, [lower_upper], hard = True)

opt = Optimizer(costs = [distanceCost, clusterCost], 
                cost_weights = [1.0, 0.2], 
                constraints = [nclustConstraint],
                modifier = ClusterSwapSmartTask(task_distance_matrix),
                num_iters = 2000,
                inner_loop_iters = 500,
                energy_plot_path='task_kmedoids_2.png'
            )

result = opt.optimize(cluster_assigns)
result_mapped = [[node_map[n] for n in r] for r in result]

print("result:", result_mapped)
print("optimized distance: ", distanceCost.eval(result))
print("optimized number of clusters: ", len(result))

plot_graph_with_cluster_ids(G, result_mapped, vary_sizes=True, color_skew_factor=20)
