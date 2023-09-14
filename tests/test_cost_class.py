import numpy as np
from opt.model.modifier import ModifierFn
from opt.model.optimizer import Optimizer
from opt.model.cost import CostFn

N = 5 # small Ising model with 5 Nodes

def test_evaluate_cost_no_args_no_kwargs_no_encoder():
    """
    Unit test for evaluating the `eval` method of the CostFn class when you pass no `cost_args`, no `cost_kwargs`, 
    and no encoder function or associated args/kwargs
    """

    J = np.random.rand(N, N) #coupling matrix

    def ising_cost(state):
        """
        Linear Hamiltonian that defines an Ising model
        """
        return -0.5 * (state.T @ J @ state)

    state = (np.random.rand(N) > 0.5).astype(np.float64) # sample a random state

    function_eval = ising_cost(state)

    IsingCost = CostFn(ising_cost)
    class_method_eval =  IsingCost.eval(state)
    assert function_eval == class_method_eval


def test_evaluate_cost_with_args_no_encoder():
    """
    Unit test for evaluating the `eval` method of the CostFn class when you pass in `cost_args` but no `cost_kwargs`,
    and no encoder function or associated args/kwargs
    """

    J = np.random.rand(N, N) #coupling matrix

    def ising_cost(state, J):
        """
        Linear Hamiltonian that defines an Ising model
        """
        return -0.5 * (state.T @ J @ state)

    state = (np.random.rand(N) > 0.5).astype(np.float64) # sample a random state

    function_eval = ising_cost(state, J)

    cost_args = [J] 
    IsingCost = CostFn(ising_cost, cost_args)
    class_method_eval =  IsingCost.eval(state)
    assert function_eval == class_method_eval


def test_evaluate_cost_with_args_and_kwargs_no_encoder():
    """
    Unit test for evaluating the `eval` method of the CostFn class when you pass in `cost_args` and `cost_kwargs`,
    and no encoder function or associated args/kwargs
    """

    J = np.random.rand(N, N) #coupling matrix, ferromagnetic

    def ising_cost(state, J, antiferro = False):
        """
        Linear Hamiltonian that defines an Ising model, with keyword argument
        that turns all interactions negative (makes thing 'antiferromagnetic)
        """
        if antiferro:
            J = -J
        return -0.5 * (state.T @ J @ state)

    state = (np.random.rand(N) > 0.5).astype(np.float64) # sample a random state

    function_eval_ferro = ising_cost(state, J, antiferro=False)

    cost_args = [J] 
    cost_kwargs = {'antiferro': False}
    IsingCost = CostFn(ising_cost, cost_args, cost_kwargs)
    class_method_eval_ferro =  IsingCost.eval(state)
    assert function_eval_ferro == class_method_eval_ferro

    function_eval_antiferro = ising_cost(state, J, antiferro=True)
    cost_kwargs = {'antiferro': True}
    IsingCost = CostFn(ising_cost, cost_args, cost_kwargs)
    class_method_eval_antiferro =  IsingCost.eval(state)
    assert function_eval_antiferro == class_method_eval_antiferro


def test_evaluate_cost_with_args_and_kwargs_encoder_fn():
    """
    Unit test for evaluating the `eval` method of the CostFn class when you pass in `cost_args` and `cost_kwargs`,
    and a simple encoder function (no associated args or kwargs) that just converts the state from a list to numpy array
    """

    J = np.random.rand(N, N) #coupling matrix, ferromagnetic

    def ising_cost(state, J, antiferro = False):
        """
        Linear Hamiltonian that defines an Ising model, with keyword argument
        that turns all interactions negative (makes thing 'antiferromagnetic)
        """
        if antiferro:
            J = -J
        return -0.5 * (state.T @ J @ state)

    state = (np.random.rand(N) > 0.5).astype(np.float64) # sample a random state

    function_eval_ferro = ising_cost(state, J, antiferro=False)

    cost_args = [J] 
    cost_kwargs = {'antiferro': False}
    encoder_fn_handle = lambda x: np.array(x) # convert from a list to a numpy array

    IsingCost = CostFn(ising_cost, cost_args, cost_kwargs, encoder_fn_handle=encoder_fn_handle)

    list_state = [node_i for node_i in state]
    class_method_eval_ferro =  IsingCost.eval(list_state)
    assert function_eval_ferro == class_method_eval_ferro
