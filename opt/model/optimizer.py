from tqdm import tqdm

from typing import List, Callable, Union
from functools import partial

from opt.utils import extract_kwargs, complement, cprint
from opt.visual.visual import plot_energy_data
from opt.model.energy import EnergyFn
from opt.model.cost import CostFn
from opt.model.modifier import ModifierFn
from opt.model.constraint import ConstraintFn
from opt.defaults.step import MH_step
from opt.defaults.scheduler import annealing_schedule_simple
from opt.defaults.cost import GraphDistanceCost, default_cost_weights
from opt.defaults.modifier import geneticMutations # I'm using the `camelCase` convention for class instances here -- 
                                                   # don't think that's very Pythonic, open to suggestions
from opt.defaults.optimizer import DEFAULT_MAX_RETRIES, DEFAULT_NUM_ITERS, DEFAULT_STEP_ITERS, DEFAULT_MAX_SUCCESSES_PER_STEP



# TODO : Get working on (generalized) k-medoids
# TODO : Check whether init procedure for Optimizer state used by scheduler_fn makes sense
# TODO : Go over scheduler_fn generally
# TODO : Check whether `inner loop` of Optimizer step should be delegated to `step_fns` (I see arguments pro and con)
# TODO : Define Annealer and Genetic subclasses
# TODO : Get this running on snaking with start + end constraints
# TODO : Get this running on reslotting

class Optimizer():
    """
    Optimizer object that can be used to find solutions of a multi-objective constrained problem using a desired iterative, 
    discrete time algorithm (e.g. Metropolis Hastings) that minimizes a global energy function.
    """
    def __init__(self,
                costs: List[CostFn],           
                cost_weights: List[float]=[], # weights associated with each cost
                constraints: List[ConstraintFn]=[],
                modifier: ModifierFn=geneticMutations,
                step_fn: Callable=MH_step,
                scheduler_fn: Callable=annealing_schedule_simple,
                max_retries: int=DEFAULT_MAX_RETRIES, # I think there's a case for including params like this that any optimizer might need in the signature
                num_iters: int=DEFAULT_NUM_ITERS, # Same
                inner_loop_iters: int=DEFAULT_STEP_ITERS, # Not sure about this -- TODO: see if it would work to delegate the inner loop to the `step_fn`
                max_successes: int=DEFAULT_MAX_SUCCESSES_PER_STEP, # This is also an inner-loop-specific thing
                energy_plot_path: Union[None, str]=None,
                verbose: bool=True,
                **optim_hyperparams  # other keyword arguments that act as algorithm-specific hyperparameters       
        ):
        """
        Initializer or constructor for an instance of the Optimizer class

        Arguments:
        ==========
        `costs` [List[CostFn]]: list of `CostFn` instances that are used to construct the energy function of the Optimizer (`self.Energy`)
        `cost_weights` [List[float]]: list of scalar weights used to combine the costs in the evaluation of the energy function (`self.Energy`)
        `constraints`: [List[ConstraintFn]]: list of `ConstraintFn` instances that are encoded as hard or soft constraints for the optimization
        `modifier`: [ModifierFn]: an instance of `ModifierFn` that is used to generate new solutions during an optimization run.
        `step_fn`: [Callable]: a callable that is used to update the state according to self.Energy.
        `scheduler_fn`: [Callable]: a function that takes `optimizer_state, shared_state, update_params` as arguments plus some hyperparameters specific to the `scheduler_fn`. It is a scheduler on hyperparameters like temperature, learning rate, etc.
        `max_retries` [int]: number of re-runs of the main optimizer loop that are allowed before "giving up" / accepting the best solution available.
        `num_iters` [int]: Number of iterations run in the main optimization loop
        `inner_loop_iters` [int]: Number of iterations used within the `step_fn` loop (i.e. how many times proposals are generated and then accepted/rejected using `self.step_fn`)
        `max_successes` [int]:
        `energy_plot_path` [str or None]:
        `verbose` [Bool]:
        """
        cost_weights = default_cost_weights(len(costs)) if len(cost_weights) == 0 else cost_weights
        self.Energy = EnergyFn(costs, cost_weights) # initialize EnergyFn class instance using provided `costs` and `cost_weights`
        self.Modifier = modifier                    # set Modifier class instance as property of `self`
        self.Constraints = constraints              # list of Constraint classes correspond to each of the constraints 

        self.step_fn = partial(step_fn, **{'energyClass': self.Energy, 'modifierClass': self.Modifier})
        # self.update_fn = partial(update_fn, **self.update_params) # ALEX: Changed how this works because we don't generally want to apply *all* 
        # the step_params (orig. called `algo_params`) once at the start of optimization -- for example `T` needs to be passed in at each 
        # step of annealing
       
        self.max_retries = max_retries
        self.num_iters = num_iters
        self.num_step_iters = inner_loop_iters
        self.max_successes_per_step = max_successes
        self.cprint = partial(cprint, condition=verbose) # This is just a cute function to print if verbose == True
        self.verbose = verbose
        self.plot_energy = False
        if energy_plot_path is not None:
            self.plot_energy = True
            self.plot_path = energy_plot_path

        self.parse_hyperparameters(scheduler_fn, optim_hyperparams)
        self.scheduler_fn = partial(scheduler_fn, **self.scheduler_params)

    def reset_trackers(self):
        """
        Reset properties that store information that you're interested in tracking over the course of an optimization run
        """
        self.energies = []
        self.state = None # These are 'state variables' for the Optimizer itself, as distinct from the `state` being optimized
        self.shared_state = None # These are more Optimizer 'state variables', but ones that need to be input to the `step_fn`.
        # For example for simple annealing, `self.shared_state` contains the temperature (`T`) variable, and `self.state` is empty.
        # For thermodynamic annealing, `self.shared_state` contains `T` and `self.state` contains `delta_CT` and `delta_ST`, which are
        # variables that the scheduler step needs but that the step function doesn't care about.

        # See `self.scheduler_step()` below

        return

    def reset_stepwise_trackers(self):
        """
        Reset properties that store information that you're interested in tracking over the course of a single optimization step
        """
        self.num_successes = 0

        return

    def parse_hyperparameters(self, scheduler_fn_handle, optim_hyperparameters):
        """
        Parse hyperparameters into those that are specific to the optimization algorithm used (the thing that parameterizes `self.step()`)
        and those that have to do with run-time parameter (e.g. `num_iters`, etc.)
        """

        self.step_params = extract_kwargs(self.step_fn, optim_hyperparameters) # extracts and sets-as-properties algorithm-specific hyperparameters
        self.scheduler_params = extract_kwargs(scheduler_fn_handle, optim_hyperparameters)
        self.run_params = complement(optim_hyperparameters, self.step_params | self.scheduler_params) # extracts and sets-as-properties general Optimizer() run-time trackers.
    
    def scheduler_step(self, *state_update_params):
        """
        Update optimizer state (e.g. annealing temperature) each iteration
        """

        # Here, `state_update_params` are states returned by the step function that are needed to update the Optimizer state.
        # For example, the energy gap in the MH_step and whether the proposal was accepted are needed for thermodynamic annealing.
        # I thought it was better to do it this way since the step function doesn't have to take the entire Optimizer as input
        # and modify its state -- though maybe this is exactly what it should do?

        # Note on initialization: the Optimizer needs to begin in some state (specified by hyperparams like `initial_temp`).
        # I sort of hacked this in at the moment by initializing these state vars to `None` and then writing the `scheduler_fn`s so that 
        # they parse these None initial states into the appropriate values which they receive as kwargs. 
        # I was trying to avoid a whole SchedulerFn class here, but maybe such a class with its own step() and initialize() 
        # functions makes more sense

        self.state, self.shared_state = self.scheduler_fn(
            self.state, self.shared_state, state_update_params, **self.scheduler_params
        )
            
    def step(self, state):
        """
        Run the step_fn for the number of iterations specified in the class definition and log the energy
        """

        self.reset_stepwise_trackers()
        for _ in range(self.num_step_iters):
            # This is the inner loop I was talking about -- a reason for putting this in the Optimizer class itself rather than
            # offloading it to the step_fn is that logging the energy will be a pretty general-purpose thing that should be in the
            # Optimizer, but then you want to keep track of the energies across the inner loop as well, so we'd want any step_fn
            # to return a list of energies, even if there's only one "inner loop" iteration. So may as well set it to 1 if not used.
            
            # Not sure how general the `num_successes` formulation is here, but I think something like this (a convergence check, basically)
            # should be included in this class
            state, energy, success, *state_update_params = self.step_fn(state, self.shared_state, **self.step_params)
            self.energies.append(energy)
            self.num_successes += success

            if self.num_successes > self.max_successes_per_step:
                break
        
        return state, self.num_successes == 0, *state_update_params

    def optimize(self, state):
        """
        Optimization run that iteratively calls `self.step(state)` to update & store the state over course of optimization
        """

        num_attempts = 0
        run_once = False

        while (num_attempts <= self.max_retries) and ((not run_once) or (not self.hard_constraints_met(state))):
            if run_once:
                self.cprint(f"Optimized state: {state}")
                self.cprint(f"Constraints not met. Retrying. Attempt {num_attempts}/{self.max_retries}")
            self.reset_trackers()
            self.scheduler_step()
            iter_range = tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)
            for _ in iter_range:
                state, success, *state_update_params = self.step(state)
                self.scheduler_step(*state_update_params)
                if success:
                    break

            run_once = True
            num_attempts += 1


        if self.plot_energy:
            self.plot_energies()
        return state

    def hard_constraints_met(self, state):
        """
        Checks if all hard constraints are met
        """

        constraint_evals = [constr.eval(state) for constr in self.Constraints if constr.hard]
        return all(constraint_evals)
    
    def plot_energies(self):
        plot_energy_data(self.energies, save_path=self.plot_path)
    

class Annealer():
    pass


class Genetic():
    pass
