from typing import List, Callable

class EnergyFn():
    """
    This class wraps the evaluation of an energy function, which is cast as a weighted sum of a set of "cost functions,", which are
    any arbitrary scalar output functions that evaluates the cost of some common `state`. These cost functions are provided as a list of Callables
    to the constructor of the `EnergyFn` instance, as well as a list of floats that correspond to the weight with which these costs should be combined
    to compute the total energy.
    """

    def __init__(self,
                costs: List[Callable],       # list of cost-function classes (each of whose `self.eval()` methods takes the same input to `calc_energy`)
                cost_weights: List[float] # list of weights associated to each cost function
        ):
        """
        Initializer or constructor for an instance of the EnergyFn class

        Arguments:
        ==========
        `costs` [List[Callable]]: list of Callables (often, instances of the `CostFn` class) that take the same argument `state`
        `cost_weights` [List[float]]: list of floats that are used to weight each of the cost functions before summing them together during evaluation (i.e. `self.calc_energy`)
        """

        assert len(costs) == len(cost_weights), "Number of cost functions must match number of associated weights!"
        
        self.costs = costs
        self.cost_weights = cost_weights
        
    def calc_energy(self, state):
        """
        Calculates the current energy of a configuration by computing a weighted sum of costs evaluated on the `state` input
        """

        weighted_costs = [weight*cost.eval(state) for (cost, weight) in zip(self.costs, self.cost_weights)]
        return sum(weighted_costs)
