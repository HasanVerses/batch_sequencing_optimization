import numpy as np
import copy
from opt.model.optimizer import Optimizer
from opt.model.cost import CostFn
from opt.model.modifier import ModifierFn



def MSE(ref, state):
    return np.sum((ref - state)**2)

def step(x):
    delta = np.random.choice([-1, 1])
    return x + delta

def step_vec(v):
    dim = np.random.choice(v.size)
    delta = np.random.choice([-1, 1])
    u = copy.deepcopy(v)
    u[dim] += delta
    return u

def test_optimizer_regression_single_variable():

    state = 10
    target = 3

    mseCost = CostFn(cost_fn_handle=MSE, cost_args=[target])
    mutation = ModifierFn(mod_fn_handles=[step], mod_probs=[1.0])

    opt = Optimizer(
        costs = [mseCost], cost_weights=[1.0],
        modifier=mutation
    )

    result = opt.optimize(state)
    assert 3 == result


def test_optimizer_regression_multi_variable():

    state = np.random.choice(11, size=10)
    target = np.random.choice(4, size=10)

    mseCost = CostFn(cost_fn_handle=MSE, cost_args=[target])
    mutation = ModifierFn(mod_fn_handles=[step_vec], mod_probs=[1.0])

    opt = Optimizer(
        costs = [mseCost], cost_weights=[1.0],
        modifier=mutation,
    )

    print(state)
    
    result = opt.optimize(state)
    print(result)

    assert np.all(target == result)

