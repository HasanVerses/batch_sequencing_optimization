from app.graph import random_graph
from app.tasks import random_task
from app.defaults.cost import GraphClusterCost, SizeDistCost_PerBin
from app.model.optimizer import Optimizer
from app.model.sequence import Sequence
from app.algorithms.kmedoids import initialize_assignments
from app.model.constraint import ConstraintFn 
from app.defaults.modifier import clusterSwap 
from app.defaults.constraint import min_cluster_size, exponential_size_dist
from app.visual.visual import plot_graph_with_cluster_ids, plot_overlaid_assignment_histograms

import networkx as nx

G = random_graph(256) # create randomly locally-connected rectangular spatial graph
distanceCost = GraphClusterCost(G)

num_clusters, min_size, max_size = 10, 2, G.number_of_nodes()

# create fa fake `batch_size_dist` that would come from e.g. NRI
batch_size_dist = exponential_size_dist(min_batch_size=min_size, max_batch_size=max_size, loc = min_size, scale = 10.0)

# `num_clusters` used to both initialize the number of initial clusters ...
cluster_assigns = initialize_assignments(G, max_k = num_clusters) 

# ... and also to build as a hard constraint that will be checked by the optimizer 
minClustSize = ConstraintFn(min_cluster_size, [min_size], hard = True)

perBin_HistCost = SizeDistCost_PerBin(batch_size_dist, p = 1.0)

opt = Optimizer(costs = [distanceCost, perBin_HistCost], 
                cost_weights = [1.0, 1.0], 
                constraints = [minClustSize],
                modifier = clusterSwap,
                num_iters = 200,
                inner_loop_iters = 300
            )

result = opt.optimize(cluster_assigns)
print("result:", result)
print("optimized distance: ", distanceCost.eval(result))

plot_overlaid_assignment_histograms(batch_size_dist, result, bin_axis_offset = 0.2, bar_width = 0.4)

# print("optimized vector of batch sizes: ", [len(assign_i) for assign_i in result])

# plot_graph_with_cluster_ids(G, result)


