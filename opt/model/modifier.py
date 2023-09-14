import numpy as np
from typing import List, Callable
from opt.defaults.cost import default_cost_weights


class ModifierFn():
    """
    Class wrapper for storing functions that modify or mutate input states using one of a finite set of possible mutation functions. The
    `self.modify()` method of this class modifies the input state using a single one of the functions in `mod_fn_handles`, and samples stochastically
    which modification function to use according to its corresponding modification probability (`mod_probs[i]`). 
    """
    
    def __init__(
        self,           
        mod_fn_handles: List[Callable],       # list of function handles corresponding to the available mutation/modification functions
        mod_probs: List[float]=[],               # list of probabilities corresponding to the probability of using each of the available modification functions
        mod_fn_args = None,                   # non-keyword arguments for each of the functions in `mod_fn_handles`
        mod_fn_kwargs = None,                 # keyword arguments for each of the functions in `mod_fn_handles`
    ):

        """ 
        Initializer or constructor for an instance of the `ModifierFn` class

        Arguments:
        ==========
        `mod_fn_handles` [List[Callable]]: list of functions that modify their inputs or return a modified version of their input
        `mod_probs` [List[float]]: list of optional ordered arguments to `cost_fn_handle`, stored in the order to which they are fed as arguments to `cost_fn_handle`.
        `mod_fn_args` [List[List] or None]: list of lists of optional ordered arguments to each of the functions in `mod_fn_handles`, stored in the order to which they are fed as arguments to each of the functions in `mod_fn_handles`.
        `mod_fn_kwargs` [List[Dict] or None]: list of dicts of keyword arguments to each of the functions in `mod_fn_handles`, where each list is stored in the same order as the function in `mod_fn_handles` that they parameterize. Each dict is stored in format of `{"kwarg_name": kwarg_value}`
        """

        mod_probs = default_cost_weights(len(mod_fn_handles)) if len(mod_probs) == 0 else mod_probs
        assert len(mod_fn_handles) == len(mod_probs), "Number of mutation functions must match number of associated sampling probabilities"
        

        self.mod_fn_handles = mod_fn_handles

        if (mod_fn_args != None):
            assert len(mod_fn_handles) == len(mod_fn_args), "Number of argument lists does not match number of provided modification functions!"

            for (ii, arg) in enumerate(mod_fn_args):
                if not isinstance(arg, list):
                    mod_fn_args[ii] = [arg] # wrap it in a list if it's not in one already

            self.mod_fn_args = mod_fn_args
        elif mod_fn_args == None:
            self.mod_fn_args = len(mod_fn_handles) * [[]]
        
        if (mod_fn_kwargs != None):
            assert len(mod_fn_handles) == len(mod_fn_kwargs), "Number of provided keyword-argument dicts does not match number of provided modification functions!"
            self.mod_fn_kwargs = mod_fn_kwargs
        elif mod_fn_kwargs == None:
            self.mod_fn_kwargs = len(mod_fn_handles) * [{}]

        if isinstance(mod_probs, (list, tuple, np.ndarray)):
            self.mod_probs = mod_probs
        else:
            raise TypeError(
                'mod_probs must be a list, tuple or 1-D numpy array'
            )

    def modify(self, state):
        """
        Randomly selects of the mutation functions and uses it to modify the current state
        """

        mod_i = np.random.choice(len(self.mod_fn_handles), p = self.mod_probs) # which modification function to apply
        modified_state = self.mod_fn_handles[mod_i](state, *self.mod_fn_args[mod_i], **self.mod_fn_kwargs[mod_i])
        # modified_state = np.random.choice(self.mod_fn_handles, p = self.mod_probs)(state)
        
        return modified_state
