from warnings import warn

def identity_fn(x, *args, **kwargs):
    return x 

class ConstraintFn():

    def __init__(self,
                constraint_fn_handle, # function handle that evaluates constraint (either upon state or upon `cost_inst`'s `self.eval()` method)
                constraint_args = None,      # non-keyword arguments for `constraint_fn_handle`
                constraint_kwargs = None,    # keyword for `constraint_fn_handle`
                encoder_fn_handle = None,    # function handle that transforms the state first into the input format needed to evaluate constraint
                encoder_args = None,         # non-keyword arguments for `encoder_fn_handle`
                encoder_kwargs = None,       # keyword arguments for `encoder_fn_handle`
                cost_inst = None,     # (optional) instance of CostFn class that evaluates the cost in case that constraint is a function of the cost
                hard = True          # Boolean flag telling whether this is a hard or non-hard constraint
            ):
        """
        Initialize an instance of the Constraint class
        """

        self.constraint_fn = constraint_fn_handle
        self.constraint_args = constraint_args or []
        self.constraint_kwargs = constraint_kwargs or {}

        self.encoder_fn = encoder_fn_handle or identity_fn
        self.encoder_args = encoder_args or []
        self.encoder_kwargs = encoder_kwargs or {}

        if (cost_inst != None ) and (encoder_fn_handle != None):
            warn(
                   "encoder_fn_handle detected as input, \
                    while cost_inst also detected as input. Input encoding is already \
                    done by `eval()` method of cost_inst, thus the provided encoder_fn_handle will not be used...\n "
                )

        if encoder_fn_handle == None and (encoder_args != None or encoder_kwargs != None):
            raise ValueError("Cannot provide encoder_args and encoder_kwargs, when encoder_fn_handle is None/not provided")

        self.Cost = cost_inst

        self.hard = hard
    
    def eval(self, state):
        """
        Evaluate the constraint, which will either be a function of the cost (of the state) or a function of the state directly
        """

        if self.Cost is None: # self.constraint_fn operates directly on the state of the system
            encoded_input = self.encoder_fn(state, *self.encoder_args, **self.encoder_kwargs)
            constraint_out = self.constraint_fn(encoded_input, *self.constraint_args, **self.constraint_kwargs)
        else:                # self.constraint_fn operates on the output of the Cost class's eval() method
            constraint_out = self.constraint_fn(self.Cost.eval(state), *self.constraint_args, **self.constraint_kwargs)

        return constraint_out
