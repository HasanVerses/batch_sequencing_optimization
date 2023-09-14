from opt.graph import random_graph
from opt.tasks import random_task
from opt.defaults.cost import GraphClusterCost, AssignmentManhattanCost 
from opt.model.optimizer import Optimizer
from opt.model.sequence import Sequence
from opt.algorithms.kmedoids import initialize_assignments
from opt.model.constraint import ConstraintFn 
from opt.defaults.modifier import clusterSwap 
from opt.defaults.constraint import min_cluster_size, max_k_equals
from opt.visual.visual import plot_graph_with_cluster_ids

import networkx as nx

# create a fake `assignment_size_vector` that would come from e.g. NRI
assignment_size_vector = 20 * [2] + 11 * [3] + 5 * [5] + 2 * [7] + [8]

G = random_graph(sum(assignment_size_vector)) # create randomly locally-connected rectangular spatial graph
distanceCost = GraphClusterCost(G)

num_clusters, min_size = len(assignment_size_vector), 2

# `num_clusters` used to both initialize the number of initial clusters ...
cluster_assigns = initialize_assignments(G, max_k = num_clusters+1) 

# ... and also to build as a hard constraint that will be checked by the optimizer 
minClustSize = ConstraintFn(min_cluster_size, [min_size], hard = True)
nclustConstraint = ConstraintFn(max_k_equals, [num_clusters], hard = True)

ManhattanCost = AssignmentManhattanCost(assignment_size_vector)

opt = Optimizer(
                costs = [distanceCost, ManhattanCost], 
                cost_weights = [1.0, 1.0], 
                constraints = [minClustSize, nclustConstraint],
                modifier = clusterSwap,
                num_iters = 200,
                inner_loop_iters = 300
            )

result = opt.optimize(cluster_assigns)
print("result:", result)
print("optimized distance: ", distanceCost.eval(result))
print("optimized vector of batch sizes: ", [len(assign_i) for assign_i in result])

plot_graph_with_cluster_ids(G, result)


