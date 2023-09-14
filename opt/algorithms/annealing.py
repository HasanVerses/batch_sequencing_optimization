import sys
import numpy as np
from tqdm import tqdm

from opt.algorithms.genetic import reverse, exchange
from opt.algorithms.common import energy, init_cost_fns, null_result, parse_results

from opt.visual.visual import plot_energy_data, live_plot_annealing_data



DEFAULT_SETTINGS = {
    'constraint_weight': 2.0,
    'algorithm': "annealing",
}

TSA_SETTINGS = {
    'constraint_weight': 3.0,
    'algorithm': 'thermodynamic',
    'max_retries': 0,
    'n': 8
}


def metropolis(energy_new, energy_s, T, epsilon=1e-12):
    """
    Metropolis criterion: accept new state if lower energy or with a probability 
    proportional to energy difference if not
    """
    if energy_new < energy_s:
        return True
    p = np.exp(-(energy_new - energy_s) / (T + epsilon))
    return np.random.choice(2, p=[1-p, p])


def anneal_step(
    graph, 
    start, 
    end, 
    current_path, 
    T, 
    num_iters, 
    max_successes, 
    reverse_prob,
    decoder,
    decoder_kwargs,
    constraint_fn, 
    constraint_penalty, 
    constraint_fn_kwargs,
    cost_fn, 
    cost_penalty, 
    cost_fn_kwargs,
):
    """Inner loop for one iteration (temperature setting) of simulated annealing"""
    num_successes = 0

    energies = []
    gaps = []
    for _ in range(num_iters):
        to_reverse = np.random.choice(2, p=[1 - reverse_prob, reverse_prob])
        if to_reverse:
            modified_path = reverse(current_path)
        else:
            modified_path = exchange(current_path)
        
        new_energy = energy(
            graph, 
            modified_path, 
            start, 
            end, 
            decoder, 
            decoder_kwargs, 
            constraint_fn, 
            constraint_penalty, 
            constraint_fn_kwargs,
            cost_fn, 
            cost_penalty, 
            cost_fn_kwargs
        )

        old_energy = energy(
            graph, 
            current_path, 
            start, 
            end, 
            decoder, 
            decoder_kwargs, 
            constraint_fn, 
            constraint_penalty, 
            constraint_fn_kwargs,
            cost_fn, 
            cost_penalty, 
            cost_fn_kwargs
        )

        gap = new_energy - old_energy

        if metropolis(new_energy, old_energy, T):
            current_path = modified_path
            old_energy = new_energy
            num_successes += 1
        
        energies.append(old_energy)
        gaps.append(gap)
            
        if num_successes > max_successes:
            break
    
    return num_successes == 0, current_path, energies, gaps


def anneal(
    graph, 
    locations, 
    start=None, 
    end=None, 
    initial_temp=0.5, 
    alpha=0.9, 
    num_steps=None,
    max_tries_at_temp_factor=100, 
    max_success_at_temp_factor=10, 
    reverse_prob=0.5,
    max_retries=100,
    initial_state=None,
    store_best=False,
    plot_energies=False,
    encoder=None,
    encoder_kwargs=None,
    decoder=None,
    decoder_kwargs=None,
    constraint_fn=None,
    constraint_penalty=100,
    constraint_fn_kwargs=None,
    cost_fn=None,
    cost_penalty=0.2,
    cost_fn_kwargs=None,
):
    """
    Simulated annealing (given constraints) with a simple exponential temperature decrease
    """

    constraint_fn, cost_fn, constraint_fn_kwargs, cost_fn_kwargs = init_cost_fns(
        constraint_fn, cost_fn, constraint_fn_kwargs, cost_fn_kwargs)

    if encoder is not None:
        if encoder_kwargs is None:
            encoder_kwargs = dict()    
        locations = encoder(*locations, **encoder_kwargs)
    
    if decoder is not None and decoder_kwargs is None:
        decoder_kwargs = dict()

    """ Search longer in spaces with higher compleixty """    
    num_nodes = len(locations) + (start is not None) + (end is not None)
    num_iters = num_nodes*max_tries_at_temp_factor
    max_successes = num_nodes*max_success_at_temp_factor
    current_path = initial_state or locations
    T = initial_temp
    if not num_steps:
        num_steps = num_nodes*4
    
    energies = []
    energy_gaps = []

    if store_best:
        best_state, best_energy = current_path, energy(
                graph, 
                current_path, 
                start, 
                end, 
                decoder, 
                decoder_kwargs, 
                constraint_fn, 
                constraint_penalty, 
                constraint_fn_kwargs,
                cost_fn,
                cost_penalty,
                cost_fn_kwargs
            )

    for _ in tqdm(range(num_steps)):
        success, current_path, step_energies, step_gaps = anneal_step(
            graph, 
            start, 
            end, 
            current_path, 
            T, 
            num_iters, 
            max_successes, 
            reverse_prob,
            decoder,
            decoder_kwargs,
            constraint_fn,
            constraint_penalty,
            constraint_fn_kwargs,
            cost_fn,
            cost_penalty,
            cost_fn_kwargs)

        if store_best:
            current_energy = step_energies[-1]
            if all(constraint_fn(current_path, **constraint_fn_kwargs)) and current_energy < best_energy:
                print("Updating best state")
                best_state, best_energy = current_path, current_energy 

        if success:
            break

        T = T*alpha

        if plot_energies:
            energies += step_energies
            energy_gaps += step_gaps

    if store_best:
        print("using best")
        current_path = best_state
    
    raw_answer, answer, distance = parse_results(current_path, start, end, graph, decoder, decoder_kwargs)

    if not all(constraint_fn(current_path, **constraint_fn_kwargs)):
        if max_retries == 0:
            return null_result(f"\nMaximum number of retries exceeded without a valid result!")
        else:
            print(f"Invalid result. Rerunning annealing ({max_retries} retries left)...")
            raw_answer, answer, distance = anneal(
                graph, 
                locations, 
                start, 
                end, 
                initial_temp,
                alpha,
                num_steps,
                max_tries_at_temp_factor,
                max_success_at_temp_factor,
                reverse_prob,
                max_retries - 1,
                initial_state,
                store_best,
                plot_energies,
                encoder,
                encoder_kwargs,
                decoder,
                decoder_kwargs,
                constraint_fn,
                constraint_penalty,
                constraint_fn_kwargs,
                cost_fn,
                cost_penalty,
                cost_fn_kwargs
            )
    
    if plot_energies:
        plot_energy_data(energies)
        plot_energy_data(energy_gaps, "Energy gap", save_path='gaps.png')

    return raw_answer, answer, distance

def anneal_class_version(
    initial_state, # initial configuration of the state of the system to be optimized
    energy_fn,     # energy class (contains all the encoder/decoder/constraint/cost function stuff)
    stepper_fn,    # stepper class (mutates or modifies the current configuration)
    initial_temp=0.5,
    alpha=0.9,
    num_steps=None,
    max_tries_at_temp_factor=100,
    max_success_at_temp_factor=10,
    max_retries=100,
    constraint_fn=None,
    constraint_penalty=100,
    constraint_fn_kwargs=None,
    store_best=False,
    plot_energies=False
):
    """
    Simulated annealing (given constraints) with a simple exponential temperature decrease
    """

    """ Stuff like this, where variables like `num_iters` and `max_successes` are made to depend on the
    the initial state or some aspect of the input, should be done outside the function -- being done somewhere in core,
    algorithm kwargs (hyperparameters) get adjusted to compensate for input complexity"""
    # num_nodes = len(locations) + (start is not None) + (end is not None)
    # num_iters = num_nodes*max_tries_at_temp_factor
    # max_successes = num_nodes*max_success_at_temp_factor
    # current_path = initial_state or locations

    num_iters = max_tries_at_temp_factor
    max_successes = max_success_at_temp_factor

    T = initial_temp
    if not num_steps:
        num_steps = num_nodes*4

    energies = []
    energy_gaps = []

    current_state = initial_state

    if store_best:
        best_state, best_energy = current_path, energy_fn.calc_energy(current_state)

    for _ in tqdm(range(num_steps)):
        success, current_state, step_energies, step_gaps = anneal_step_class_version(
            current_state,
            T, 
            num_iters, 
            max_successes, 
            stepper_fn,
            energy_fn
            )

        """ Need a way to make sure the code below gels with the new class-based formulation... """
        if store_best:
            current_energy = step_energies[-1]
            if all(constraint_fn(current_path, **constraint_fn_kwargs)) and current_energy < best_energy:
                print("Updating best state")
                best_state, best_energy = current_state, current_energy 

        if success:
            break

        T = T*alpha

        if plot_energies:
            energies += step_energies
            energy_gaps += step_gaps
    
    if store_best:
        print("using best")
        current_state = best_state
    
    """ Need to work on below """
    # raw_answer, answer, distance = parse_results(current_path, start, end, graph, decoder, decoder_kwargs)

    if not all(constraint_fn(current_path, **constraint_fn_kwargs)):
        if max_retries == 0:
            return null_result()
        else:
            print(f"Invalid result. Rerunning annealing ({max_retries} retries left)...")
            raw_answer, answer, distance = anneal(
                graph, 
                locations, 
                start, 
                end, 
                initial_temp,
                alpha,
                num_steps,
                max_tries_at_temp_factor,
                max_success_at_temp_factor,
                reverse_prob,
                max_retries - 1,
                initial_state,
                store_best,
                plot_energies,
                encoder,
                encoder_kwargs,
                decoder,
                decoder_kwargs,
                constraint_fn,
                constraint_penalty,
                constraint_fn_kwargs,
                cost_fn,
                cost_penalty,
                cost_fn_kwargs
            )
    
    if plot_energies:
        plot_energy_data(energies)
        plot_energy_data(energy_gaps, "Energy gap", save_path='gaps.png')

    return raw_answer, answer, distance


def anneal_step_class_version(
    current_state,
    T, 
    num_iters, 
    max_successes, 
    stepper_fn,
    energy_fn
):
    """Inner loop for one iteration (temperature setting) of simulated annealing"""
    num_successes = 0

    energies = []
    gaps = []
    for _ in range(num_iters):
        
        old_energy = energy_fn.calc_energy(current_state)

        modified_state = stepper_fn.step(current_state)

        new_energy = energy_fn.calc_energy(modified_state)

        gap = new_energy - old_energy

        if metropolis(new_energy, old_energy, T):
            current_state = modified_state
            old_energy = new_energy
            num_successes += 1
        
        energies.append(old_energy)
        gaps.append(gap)
            
        if num_successes > max_successes:
            break
    
    return num_successes == 0, current_path, energies, gaps

        
def threshold_accept(energy_new, energy_s, T):
    energy_gap = energy_new - energy_s

    return energy_gap < T
    
def ta_step(
    graph, 
    start, 
    end, 
    current_path, 
    T, 
    num_iters, 
    max_successes, 
    reverse_prob,
    decoder,
    decoder_kwargs,
    constraint_fn, 
    constraint_penalty, 
    constraint_fn_kwargs,
    cost_fn, 
    cost_penalty, 
    cost_fn_kwargs
):
    """Inner loop for one iteration (temperature setting) of simulated annealing"""
    num_successes = 0
    for _ in range(num_iters):
        to_reverse = np.random.choice(2, p=[1 - reverse_prob, reverse_prob])
        if to_reverse:
            modified_path = reverse(current_path)
        else:
            modified_path = exchange(current_path)
        
        if threshold_accept(
            energy(
                graph, 
                modified_path, 
                start, 
                end, 
                decoder, 
                decoder_kwargs, 
                constraint_fn, 
                constraint_penalty, 
                constraint_fn_kwargs,
                cost_fn, 
                cost_penalty, 
                cost_fn_kwargs
            ), 
            energy(
                graph, 
                current_path, 
                start, 
                end, 
                decoder, 
                decoder_kwargs, 
                constraint_fn, 
                constraint_penalty, 
                constraint_fn_kwargs,
                cost_fn, 
                cost_penalty, 
                cost_fn_kwargs
            ), 
            T
        ):
            current_path = modified_path
            num_successes += 1
            
        if num_successes > max_successes:
            break
    
    return num_successes == 0, current_path


def deterministic_anneal(
    graph, 
    locations, 
    start=None, 
    end=None, 
    initial_threshold=30, 
    alpha=0.9, 
    num_steps=None,
    max_tries_at_temp_factor=100, 
    max_success_at_temp_factor=10, 
    reverse_prob=0.5,
    max_retries=100,
    initial_state=None,
    encoder=None,
    encoder_kwargs=None,
    decoder=None,
    decoder_kwargs=None,
    constraint_fn=None,
    constraint_penalty=100,
    constraint_fn_kwargs=None,
    cost_fn=None,
    cost_penalty=0.2,
    cost_fn_kwargs=None
):
    """
    Deterministic version of simulated annealing based on threshold acceptance
    """

    constraint_fn, cost_fn, constraint_fn_kwargs, cost_fn_kwargs = init_cost_fns(
        constraint_fn, cost_fn, constraint_fn_kwargs, cost_fn_kwargs)
    
    if encoder is not None:
        if encoder_kwargs is None:
            encoder_kwargs = dict()    
        locations = encoder(*locations, **encoder_kwargs)
    
    if decoder is not None and decoder_kwargs is None:
        decoder_kwargs = dict()
        
    num_nodes = len(locations) + (start is not None) + (end is not None)
    num_iters = num_nodes*max_tries_at_temp_factor
    max_successes = num_nodes*max_success_at_temp_factor
    current_path = initial_state or locations
    T = initial_threshold
    if not num_steps:
        num_steps = num_nodes*4

    for _ in tqdm(range(num_steps)):
        success, current_path = ta_step(
            graph, 
            start, 
            end, 
            current_path, 
            T, 
            num_iters, 
            max_successes, 
            reverse_prob,
            decoder,
            decoder_kwargs,
            constraint_fn,
            constraint_penalty,
            constraint_fn_kwargs,
            cost_fn,
            cost_penalty,
            cost_fn_kwargs,
            )

        if success:
            break

        T = T*alpha

    raw_answer, answer, distance = parse_results(current_path, start, end, graph, decoder, decoder_kwargs)

    if not all(constraint_fn(current_path, **constraint_fn_kwargs)):
        if max_retries == 0:
            return null_result(f"\nMaximum number of retries exceeded without a valid result!")
        else:
            print(f"Invalid result. Rerunning d-annealing ({max_retries} retries left)...")
            raw_answer, answer, distance = deterministic_anneal(
                graph, 
                locations, 
                start, 
                end, 
                initial_threshold,
                alpha,
                num_steps,
                max_tries_at_temp_factor,
                max_success_at_temp_factor,
                reverse_prob,
                max_retries - 1,
                initial_state,
                encoder,
                encoder_kwargs,
                decoder,
                decoder_kwargs,
                constraint_fn,
                constraint_penalty,
                constraint_fn_kwargs,
                cost_fn, 
                cost_penalty, 
                cost_fn_kwargs
            )

    return raw_answer, answer, distance


def anneal_thermodynamic(
    graph, 
    locations, 
    start=None, 
    end=None, 
    initial_temp=1e-2, #1e-2
    final_temp=2e-5, #2e-5
    kA=5000, #10000
    burn_in=16000, #16000
    num_steps=None,
    reverse_prob=0.5,
    max_retries=100,
    initial_state=None,
    mode="stochastic",
    start_from_best=False,
    plot_energies=False,
    live_plot=False,
    encoder=None,
    encoder_kwargs=None,
    decoder=None,
    decoder_kwargs=None,
    constraint_fn=None,
    constraint_penalty=100,
    constraint_fn_kwargs=None,
    cost_fn=None,
    cost_penalty=0.2,
    cost_fn_kwargs=None,
):
    """
    Simulated annealing (given constraints) with online thermodynamic temperature adjustment
    """
    assert mode in ["stochastic", "deterministic"], "`mode` parameter takes values `stochastic` or `deterministic`!"
    criterion = metropolis if mode == 'stochastic' else threshold_accept

    constraint_fn, cost_fn, constraint_fn_kwargs, cost_fn_kwargs = init_cost_fns(
        constraint_fn, cost_fn, constraint_fn_kwargs, cost_fn_kwargs)

    if encoder is not None:
        if encoder_kwargs is None:
            encoder_kwargs = dict()    
        locations = encoder(*locations, **encoder_kwargs)
    
    if decoder is not None and decoder_kwargs is None:
        decoder_kwargs = dict()
        
    num_nodes = len(locations) + (start is not None) + (end is not None)
    current_path = initial_state or locations
    T = initial_temp
    if not num_steps:
        num_steps = num_nodes*4
    
    energies = []
    energy_gaps = []
    temps = []

    delta_CT = 0.
    delta_ST = 0.

    if start_from_best:
        best_state, best_energy = current_path, energy(
                graph, 
                current_path, 
                start, 
                end, 
                decoder, 
                decoder_kwargs, 
                constraint_fn, 
                constraint_penalty, 
                constraint_fn_kwargs,
                cost_fn,
                cost_penalty,
                cost_fn_kwargs
            )

    k = 0
    while T > final_temp or k < burn_in:
        to_reverse = np.random.choice(2, p=[1 - reverse_prob, reverse_prob])
        if to_reverse:
            modified_path = reverse(current_path)
        else:
            modified_path = exchange(current_path)

        new_energy = energy(
            graph, 
            modified_path, 
            start, 
            end, 
            decoder, 
            decoder_kwargs, 
            constraint_fn, 
            constraint_penalty, 
            constraint_fn_kwargs,
            cost_fn, 
            cost_penalty, 
            cost_fn_kwargs
        )

        old_energy = energy(
            graph, 
            current_path, 
            start, 
            end, 
            decoder, 
            decoder_kwargs, 
            constraint_fn, 
            constraint_penalty, 
            constraint_fn_kwargs,
            cost_fn, 
            cost_penalty, 
            cost_fn_kwargs
        )

        gap = new_energy - old_energy

        if criterion(new_energy, old_energy, T):
            current_path = modified_path
            old_energy = new_energy
            delta_CT += gap
        
        if gap > 0:
            delta_ST -= gap/(T+1e-12)
        
        if delta_CT >= 0 or delta_ST == 0:
            T = initial_temp
        else:
            T = kA*(delta_CT/delta_ST)
        
        if k%500 == 0:
            sys.stdout.write("\riter: {0}, temp: {1}".format(k, T))
            sys.stdout.flush()
        
        temps.append(T)
        energies.append(old_energy)
        energy_gaps.append(gap)

        if live_plot:
            live_plot_annealing_data(temps)

        if start_from_best:
            current_energy = old_energy
            if all(constraint_fn(current_path, **constraint_fn_kwargs)) and current_energy < best_energy:
                print("Updating best state")
                best_state, best_energy = current_path, current_energy 
        
        k += 1

    raw_answer, answer, distance = parse_results(current_path, start, end, graph, decoder, decoder_kwargs)

    if not all(constraint_fn(current_path, **constraint_fn_kwargs)):
        if max_retries == 0:
            return null_result(f"\nMaximum number of retries exceeded without a valid result!")
        else:
            print(f"Invalid result. Rerunning annealing ({max_retries} retries left)...")
            raw_answer, answer, distance = anneal_thermodynamic(
                graph, 
                locations, 
                start, 
                end, 
                initial_temp,
                final_temp,
                kA,
                burn_in,
                num_steps,
                reverse_prob,
                max_retries - 1,
                best_state if start_from_best else initial_state,
                mode,
                start_from_best,
                plot_energies,
                live_plot,
                encoder,
                encoder_kwargs,
                decoder,
                decoder_kwargs,
                constraint_fn,
                constraint_penalty,
                constraint_fn_kwargs,
                cost_fn,
                cost_penalty,
                cost_fn_kwargs
            )
    
    if plot_energies:
        plot_energy_data(energies)
        plot_energy_data(energy_gaps, "Energy gap", save_path='gaps.png')

    return raw_answer, answer, distance
