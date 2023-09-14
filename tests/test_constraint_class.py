import numpy as np
from opt.model.modifier import ModifierFn
from opt.model.optimizer import Optimizer
from opt.model.cost import CostFn
from opt.model.constraint import ConstraintFn 
from opt.defaults.cost import SpinGlassCost

def test_evaluate_constraint_noArgs_noKwargs_noEncoder_noCost():
    """
    Unit test for evaluating the `eval` method of the ConstraintFN class when you pass no `constraint_args`, no `constraint_kwargs`, 
    no `encoder_fn_handle` or associated args/kwargs, and no `cost_inst`
    """

    k = 5

    def is_length_K(state):
        return (True if len(state) == k else False)

    state = [np.random.choice(5, size = 6), np.random.choice(5, size = 2)]

    constraint_eval = is_length_K(state)

    LengthK_Constraint = ConstraintFn(is_length_K)
    class_method_eval =  LengthK_Constraint.eval(state)
    assert constraint_eval == class_method_eval

def test_evaluate_constraint_withArgs_noKwargs_noEncoder_noCost():
    """
    Unit test for evaluating the `eval` method of the ConstraintFN class when you pass `constraint_args`, no `constraint_kwargs`, 
    no `encoder_fn_handle` or associated args/kwargs, and no `cost_inst`
    """

    def is_length_K(state, k):
        return (True if len(state) == k else False)

    state = [np.random.choice(5, size = 6), np.random.choice(5, size = 2)]

    target_length = 2

    constraint_eval = is_length_K(state, target_length)

    LengthK_Constraint = ConstraintFn(is_length_K, [target_length])
    class_method_eval =  LengthK_Constraint.eval(state)
    assert constraint_eval is True
    assert class_method_eval is True

    new_state = [np.random.choice(5, size = 6), np.random.choice(5, size = 2), np.random.choice(5, size = 2)]

    constraint_eval = is_length_K(new_state, target_length)
    class_method_eval =  LengthK_Constraint.eval(new_state)
    assert constraint_eval is False
    assert class_method_eval is False

def test_evaluate_constraint_withArgs_withKwargs_noEncoder_noCost():
    """
    Unit test for evaluating the `eval` method of the ConstraintFN class when you pass in `constraint_args`, `constraint_kwargs`, 
    no `encoder_fn_handle` or associated args/kwargs, and no `cost_inst`
    """

    def is_length_K(state, k, string_out = "gooberman"):
        print(f'{string_out}\n')
        return (True if len(state) == k else False), string_out

    state = [np.random.choice(5, size = 6), np.random.choice(5, size = 2)]

    target_length = 2

    constraint_eval, string_out_fneval = is_length_K(state, target_length, string_out="grub")

    constraint_args = [target_length] 
    constraint_kwargs = {'string_out': "grub"}

    LengthK_Constraint = ConstraintFn(is_length_K, constraint_args, constraint_kwargs)
    class_method_eval, string_out_classeval =  LengthK_Constraint.eval(state)
    assert constraint_eval == class_method_eval
    assert string_out_fneval == string_out_classeval

def test_evaluate_constraint_noArgs_noKwargs_noEncoder_withCost():
    """
    Unit test for evaluating the `eval` method of the ConstraintFN class when you pass no `constraint_args`, no `constraint_kwargs`, 
    no `encoder_fn_handle` or associated args/kwargs, and a `cost_inst`
    """

    J = np.ones((3,3)) - np.eye(3)
    my_cost = SpinGlassCost(J, antiferro = False)
    state = np.array([1.0, 1.0, 1.0])

    def is_negative(cost_eval):
        return True if cost_eval < 0 else False

    NegEnergyConstraint = ConstraintFn(is_negative, cost_inst=my_cost)
    energy_negative = NegEnergyConstraint.eval(state)
    assert energy_negative

    my_cost = SpinGlassCost(J, antiferro = True)
    NegEnergyConstraint = ConstraintFn(is_negative, cost_inst=my_cost)
    energy_negative = NegEnergyConstraint.eval(state)
    assert not energy_negative
