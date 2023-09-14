from opt.graph import random_graph
from opt.tasks import random_task
from opt.defaults.cost import GraphClusterCost
from opt.model.optimizer import Optimizer
from opt.model.sequence import Sequence
from opt.graph import get_distance_matrix
from opt.algorithms.kmedoids import initialize_assignments
from opt.visual.visual import plot_graph_with_cluster_ids
from opt.algorithms.kmedoids import random_move, random_swap, random_move_shortest_path_minimize
from opt.model.modifier import ModifierFn

# from functools import partial
import networkx as nx

G = random_graph(200) # create randomly locally-connected rectangular spatial graph
num_clusters = 5     # assert desired number of graph clusters (@NOTE: Should be baked into hyperparams of optimizer somehow/one-day?)
distanceCost = GraphClusterCost(G)

# random_move_ssp = partial(random_move_shortest_path_minimize, graph_distance_matrix = get_distance_matrix(G))
clusterSwapSmart = ModifierFn([random_move_shortest_path_minimize], [1.0], [ [get_distance_matrix(G)] ])

cluster_assigns = initialize_assignments(G, max_k = num_clusters)

opt = Optimizer(costs = [distanceCost], 
                cost_weights = [1.0], 
                modifier = clusterSwapSmart,
                inner_loop_iters = 300
            )

result = opt.optimize(cluster_assigns)
print("result:", result)
print("optimized distance: ", distanceCost.eval(result))

plot_graph_with_cluster_ids(G, result)

