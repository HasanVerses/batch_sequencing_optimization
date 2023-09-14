from opt.graph import random_graph
from opt.tasks import random_task
from opt.defaults.cost import GraphClusterCost, numClusterCost
from opt.model.optimizer import Optimizer
from opt.model.sequence import Sequence
from opt.algorithms.kmedoids import initialize_assignments
from opt.model.constraint import ConstraintFn 
from opt.defaults.modifier import clusterSwap 
from opt.defaults.constraint import max_k_between
from opt.visual.visual import plot_graph_with_cluster_ids

import networkx as nx

G = random_graph(200) # create randomly locally-connected rectangular spatial graph
num_clusters = 5     # start with a desired number of graph clusters
lower_upper =  [num_clusters - 2, num_clusters + 2] # allowable bounds for number of clusters
distanceCost = GraphClusterCost(G)
clusterCost = numClusterCost(num_clusters)

# `num_clusters` used to both initialize the number of initial clusters ...
cluster_assigns = initialize_assignments(G, max_k = num_clusters) 

# ... and also to build as a hard constraint that will be checked by the optimizer 
nclustConstraint = ConstraintFn(max_k_between, [lower_upper], hard = True)

opt = Optimizer(costs = [distanceCost, clusterCost], 
                cost_weights = [1.0, 0.5], 
                constraints = [nclustConstraint],
                modifier = clusterSwap,
                num_iters = 200,
                inner_loop_iters = 300
            )

result = opt.optimize(cluster_assigns)
print("result:", result)
print("optimized distance: ", distanceCost.eval(result))
print("optimized number of clusters: ", len(result))

plot_graph_with_cluster_ids(G, result)


