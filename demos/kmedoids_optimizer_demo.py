from opt.graph import random_graph
from opt.tasks import random_task
from opt.defaults.cost import GraphClusterCost
from opt.model.optimizer import Optimizer
from opt.model.sequence import Sequence
from opt.graph import get_distance_matrix
from opt.algorithms.kmedoids import initialize_assignments
from opt.defaults.modifier import clusterSwap 
from opt.visual.visual import plot_graph_with_cluster_ids

import networkx as nx

G = random_graph(200) # create randomly locally-connected rectangular spatial graph
num_clusters = 5     # assert desired number of graph clusters (@NOTE: Should be baked into hyperparams of optimizer somehow/one-day?)
distanceCost = GraphClusterCost(G)

cluster_assigns = initialize_assignments(G, max_k = num_clusters)

opt = Optimizer(costs = [distanceCost], 
                cost_weights = [1.0], 
                modifier = clusterSwap,
                inner_loop_iters = 300
            )

result = opt.optimize(cluster_assigns)
print("result:", result)
print("optimized distance: ", distanceCost.eval(result))

plot_graph_with_cluster_ids(G, result)

