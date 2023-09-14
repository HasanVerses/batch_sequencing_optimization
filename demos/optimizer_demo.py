from opt.graph import random_graph
from opt.tasks import random_task
from opt.defaults.cost import GraphDistanceCost
from opt.model.optimizer import Optimizer
from opt.model.sequence import Sequence

from opt.visual.visual import plot_graph



G = random_graph(200)
start, end, locations = random_task(G, 20)
distanceCost = GraphDistanceCost(G)

print("start:", start, "end:", end)
print("locations:", locations)
print("unoptimized distance: ", distanceCost.eval(locations))

opt = Optimizer(costs=[distanceCost])
result = opt.optimize([start] + locations + [end])

print("result:", result)
print("optimized distance: ", distanceCost.eval(result))
plot_graph(G, sequence=Sequence(result, G, use_bins=False))
