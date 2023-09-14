# class Annealer():

    # def __init__(costs,           # costs (classes)
    #             cost_weights,     # weights associated with each cost
    #             modifier,         # stepper/modifier (class)
    #             constraints,      # constraints (classes)
    #             T,                # temperature
    #             num_iters         # number of iterations
    # ):

    #     self.Energy = EnergyFn(costs, cost_weights)
    #     self.Modifier = modifier
    #     self.Constraints = constraints
    #     self.T = T
    #     self.num_iters = num_iters

    #     self.num_successes = 0

    # def check_constraints(state):

    #     which_constraints_met = [constraint.eval(state) for constraint in self.Constraints if constraint.hard_flag]
    #     return all(which_constraints_met)

    # def __init__(costs,           # costs (classes)
    #             cost_weights,     # weights associated with each cost
    #             modifier,         # stepper/modifier (class)
    #             constraint_idx,   # list of indices of the costs which are used to create hard constraints
    #             constraint_thr,   # the values of the costs with indices given by `constraint_idx` that must be crossed in order for constraint to be met
    #             T,                # temperature
    #             num_iters         # number of iterations
    # ):

    #     self.Energy = EnergyFn(costs, cost_weights)
    #     self.Modifier = modifier

    #     constraint_fns = []
    #     for (ii, cost_fn_idx) in enumerate(constraint_idx):
    #         constraint_fns.append(lambda x: costs[cost_fn_idx].eval(x) < constraint_thr[ii])
            
    #     self.constraint_fns = constraint_fns
    #     self.T = T
    #     self.num_iters = num_iters

    #     self.num_successes = 0

    # def check_constraints(self, state):

    #     which_constraints_met = [constraint_fn(state) for constraint_fn in self.constraint_fns]
    #     return all(which_constraints_met)

    # def run(self, state):

    #     current_state = state
    #     for _ in range(num_iters):
            
    #         current_energy = self.Energy.calc_energy(state)
    #         modded_state = self.Modifier.modify(current_state)
    #         new_energy = self.Energy.calc_energy(modded_state)

    #         if metropolis(new_energy, current_energy, self.T):
    #             current_state = modded_state
    #             old_energy = new_energy
    #         self.num_successes += 1
        