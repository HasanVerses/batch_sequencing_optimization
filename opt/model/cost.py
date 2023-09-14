from typing import Callable, Dict, List

def identity_fn(x, *args, **kwargs):
    return x 

class CostFn():
    """
    This class wraps the evaluation of an arbitrary scalar output function of some singular `state`, where
    the function handle is given by `cost_fn_handle` with optional arguments (`cost_args`) and
    keyword arguments (`cost_kwargs`). 
    The constructor can be provided an optional function (with function handle `encoder_fn_handle`) that "encodes" 
    or transforms the input state on which the cost function is evaluated, before evaluation. This encoder function
    can have additional optional arguments (`encoder_args`) and keyword arguments (`encoder_kwargs`).
    
    =============
    Usage example using the full functionality of the `CostFn` class:
    =============
    ```
    def ising_cost(state, J, antiferro = False):
        if antiferro:
            J = -J
        return -0.5 * (state.T @ J @ state)

    state = (np.random.rand(N) > 0.5).astype(np.float64) # sample a random state

    J = np.random.randn(N, N)
    cost_args = [J] 
    cost_kwargs = {'antiferro': False}
    encoder_fn_handle = lambda x: np.array(x) # convert from a list to a numpy array
    IsingCost = CostFn(ising_cost, cost_args, cost_kwargs, encoder_fn_handle=encoder_fn_handle)
    ```
    """

    def __init__(self,
                cost_fn_handle: Callable, 
                cost_args: List = None,
                cost_kwargs: Dict = None, 
                encoder_fn_handle: Callable = None,
                encoder_args: List = None,
                encoder_kwargs: Dict = None,
        ):
        """ 
        Initializer or constructor for an instance of the `CostFn` class

        Arguments:
        ==========
        `cost_fn_handle` [Callable]: function that outputs a scalar, operates on a single "state" argument with additional optional `cost_args` and `cost_kwargs` (both empty/missing by default).
        `cost_args` [List]: list of optional ordered arguments to `cost_fn_handle`, stored in the order to which they are fed as arguments to `cost_fn_handle`.
        `cost_kwargs` [Dict]: dict of keyword arguments to `cost_fn_handle`, stored in format of `{"kwarg_name": kwarg_value}`
        `encoder_fn_handle` [Callable]: function that transforms the `state` input to `self.eval()` before passing it to `self.cost_fn`
        `encoder_args` [List]: list of optional ordered arguments to `encoder_fn_handle`, stored in the order to which they are fed as arguments to `encoder_fn_handle`.
        `encoder_kwargs` [Dict]: dict of keyword arguments to `encoder_fn_handle`, stored in format of `{"kwarg_name": kwarg_value}`
        """
       
        assert cost_fn_handle is not None, "Must pass in a cost function handle!"

        self.cost_fn = cost_fn_handle

        self.cost_args = cost_args or []
        self.cost_kwargs = cost_kwargs or {}

        self.encoder_fn = encoder_fn_handle or identity_fn
        self.encoder_args = encoder_args or []
        self.encoder_kwargs = encoder_kwargs or {}

    def eval(self, state):
        """
        Encodes the input using self.encoder_fn(state, ...) and evaluates the cost on the encoded input
        """

        encoded_input = self.encoder_fn(state, *self.encoder_args, **self.encoder_kwargs)

        return self.cost_fn(encoded_input, *self.cost_args, **self.cost_kwargs)
