import numpy as np
import copy

from opt.model.optimizer import Optimizer
from opt.model.cost import CostFn
from opt.model.modifier import ModifierFn



state = 10
target = 3

def MSE(ref, state):
    return (ref - state)**2

mseCost = CostFn(cost_fn_handle=MSE, cost_args=[target])

def step(x):
    delta = np.random.choice([-1, 1])
    return x + delta

mutation = ModifierFn(mod_fn_handles=[step], mod_probs=[1.0])

# opt = Optimizer(
#     costs = [mseCost], cost_weights=[1.0],
#     modifier=mutation
# )

# result = opt.optimize(state)
# print(result)

print("------Multivariate------")

def step_vec(v):
    dim = np.random.choice(v.size)
    delta = np.random.choice([-1, 1])
    u = copy.deepcopy(v)
    u[dim] += delta
    return u


state = np.random.choice(11, size=3)
target = np.random.choice(4, size=3)

mseCost = CostFn(cost_fn_handle=MSE, cost_args=[target])
mutation = ModifierFn(mod_fn_handles=[step_vec], mod_probs=[1.0])

opt = Optimizer(
    costs = [mseCost], cost_weights=[1.0],
    modifier=mutation,
    energy_plot_path='unit_test.png',
    initial_temp = 0.0001
)

print("INITIAL: ", state)
print("TARGET: ", target)

result = opt.optimize(state)
print(result)

