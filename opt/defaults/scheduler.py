DEFAULT_INITIAL_TEMP = 0.5
DEFAULT_ALPHA = 0.9
DEFAULT_KA_VALUE = 5000



def annealing_schedule_simple(optimizer_state, shared_state, update_params, alpha=DEFAULT_ALPHA, initial_temp=DEFAULT_INITIAL_TEMP):
    if shared_state is None:
        return [], [initial_temp]
    
    T, = shared_state
    return [], [T*alpha]
    

def annealing_schedule_thermodynamic(
    optimizer_state,
    shared_state, 
    update_params, 
    kA=DEFAULT_KA_VALUE, 
    initial_temp=DEFAULT_INITIAL_TEMP
):
    if optimizer_state is None or shared_state is None:
        return [0., 0.], [initial_temp]

    delta_CT, delta_ST = optimizer_state
    T, = shared_state
    E_gap, modified = update_params

    if modified:
        delta_CT += E_gap

    if E_gap > 0:
        delta_ST -= E_gap/(T+1e-12)
    
    if delta_CT >= 0 or delta_ST == 0:
        T = initial_temp
    else:
        T = kA*(delta_CT/delta_ST)
    
    return [delta_CT, delta_ST], [T]
