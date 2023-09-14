from opt.algorithms.annealing import metropolis


# TODO: GAH

# step function has to return vars to update annealer 'state' i.e. scheduler_fn (scheduler_update_vars)
# scheduler_fns have to take in 'annealer state' + scheduler_update_vars
# step function has to TAKE IN only some annealer state vars (doesn't care about others)
# scheduler function has to deal with default values but separate kwargs from args (annealer state + update vars)


def MH_step(state, optimizer_state, modifierClass=None, energyClass=None):
    """
    Example metropolis hastings function update function that can be used as the `self.update_fn()` for the Optimizer class.
    It's just a thin wrapper around `metropolis` function from app.algorithms.annealing.
    """
    assert modifierClass and energyClass, "Must supply `modifierClass` and `energyClass` (this should be handled by an Optimizer instance)!"
    assert len(optimizer_state) == 1, "Optimizer should supply only one temperature variable"

    T, = optimizer_state
    
    current_energy = energyClass.calc_energy(state)    # calculate energy of input state
    modded_state = modifierClass.modify(state)         # generate new "proposal" state (modified state)
    new_energy = energyClass.calc_energy(modded_state) # calculate energy of the modified state
    E_gap = new_energy - current_energy

    if metropolis(new_energy, current_energy, T):
        out = modded_state
        energy_out = new_energy
        modified = True
    else:
        out = state
        energy_out = current_energy
        modified = False

    return out, energy_out, modified, E_gap
