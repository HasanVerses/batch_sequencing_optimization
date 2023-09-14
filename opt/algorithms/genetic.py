import numpy as np
from tqdm import tqdm

from opt.algorithms.common import energy, init_cost_fns, null_result, parse_results



def exchange(sequence):
    """Choose three random indices and swap sequence[0:1] with sequence[1:2]"""
    length = len(sequence)
    if length == 2:
        return sequence[::-1]
    elif length == 1:
        return sequence
    first_cut = np.random.choice(length)
    second_cut = np.random.choice(length+1-first_cut) + first_cut
    third_cut = np.random.choice(length+1-second_cut) + second_cut
    
    return sequence[:first_cut] + sequence[second_cut:third_cut] + sequence[first_cut:second_cut] + sequence[third_cut:]

def reverse(sequence):
    """Choose two random indices and reverse sequence[0:1]"""

    length = len(sequence)
    if length == 2:
        return sequence[::-1]
    elif length == 1:
        return sequence
    first_cut = np.random.choice(length)
    second_cut = np.random.choice(length+1-first_cut) + first_cut
    
    return sequence[:first_cut] + sequence[first_cut:second_cut][::-1] + sequence[second_cut:]


def distance_rank(sequences, distances, top_k):
    return [x for _, x in sorted(zip(distances, sequences), key=lambda pair: pair[0])][:top_k]


def selection(
    graph, 
    start, 
    end, 
    population, 
    top_k, 
    decoder, 
    decoder_kwargs, 
    constraint_fn, 
    constraint_penalty, 
    constraint_fn_kwargs,
    cost_fn,
    cost_penalty,
    cost_fn_kwargs
):
    """
    Measure 'fitness' (shortness of path) across population and return only the top k
    """
    distances = [energy(
        graph, 
        sequence, 
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
    ) for sequence in population]
    
    return distance_rank(population, distances, top_k)


def genetic_step(population, replication_factor, exchange_prob):
    """
    Single step for genetic algorithm
    """
    new_population = []
    for sequence in population:
        for _ in range(replication_factor):
            apply_exchange = np.random.choice(2, p=[1-exchange_prob, exchange_prob])
            new_population.append(exchange(sequence) if apply_exchange else reverse(sequence))
    
    return new_population


def genetic(
    graph, 
    locations, 
    start=None, 
    end=None, 
    pop_size=20, 
    replication_factor=4, 
    max_iters=200, 
    top_k=5, 
    exchange_prob=0.7,
    initial_state=None,
    max_retries=100,
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
    Genetic algorithm based on mutations (reverse and exchange)
    """
    constraint_fn, cost_fn, constraint_fn_kwargs, cost_fn_kwargs = init_cost_fns(
        constraint_fn, cost_fn, constraint_fn_kwargs, cost_fn_kwargs)

    if encoder is not None:
        if encoder_kwargs is None:
            encoder_kwargs = dict()    
        locations = encoder(locations, **encoder_kwargs)

    if decoder is not None and decoder_kwargs is None:
        decoder_kwargs = dict()

    population = [list(np.random.permutation(locations)) for _ in range(pop_size)]
    if initial_state is not None:
        population[0] = initial_state
    for _ in tqdm(range(max_iters)):
        population += genetic_step(population, replication_factor, exchange_prob)
        population = selection(
            graph, 
            start, 
            end, 
            population, 
            top_k, 
            decoder,
            decoder_kwargs,
            constraint_fn, constraint_penalty, constraint_fn_kwargs,
            cost_fn, cost_penalty, cost_fn_kwargs)

    raw_answer, answer, distance = parse_results(population[0], start, end, graph, decoder, decoder_kwargs)        

    if not all(constraint_fn(population[0], **constraint_fn_kwargs)):
        if max_retries == 0:
            return null_result()
        else:
            print(f"Invalid solution found; re-running algorithm ({max_retries} attempts left)")
            raw_answer, answer, distance = genetic(
                graph, 
                locations, 
                start, 
                end, 
                pop_size, 
                replication_factor, 
                max_iters, 
                top_k, 
                exchange_prob,
                initial_state,
                max_retries - 1,
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


# With crossover

def conditional_replace(mapping, excerpt, parent_sequence):
    result = []
    for c in parent_sequence:
        while c in excerpt and c in mapping:
            c = mapping[c]
        result.append(c)
    
    return result


def pmx(parent_A, parent_B):
    """
    Crossover function that produces offspring by exchanging a segment between parents, then mapping 
    bits outside one of the exchanged segments to their images in the map

    i.e. 
    parent A = 0 1 2 [3 4 5] 6 7 8 9  ->  0 1 2 [6 5 4] 6 7 8 9  ->  0 1 2 [6 5 4] 3 7 8 9
    parent B = 9 8 7 [6 5 4] 3 2 1 0  ->  9 8 7 [3 4 5] 3 2 1 0  ->  9 8 7 [3 4 5] 6 2 1 0
    """
    length = len(parent_A)
    assert len(parent_A) >= 3
    first_cut = np.random.choice(length - 2)
    second_cut = np.random.choice(length - 1 - first_cut) + first_cut
    
    A_excerpt = parent_A[first_cut:second_cut]
    B_excerpt = parent_B[first_cut:second_cut]
    A_map = {B: A for A, B in list(zip(A_excerpt, B_excerpt))}
    B_map = {v: k for k, v in A_map.items()}
    
    kid_A = conditional_replace(A_map, B_excerpt, parent_A[:first_cut]) + B_excerpt + conditional_replace(A_map, B_excerpt, parent_A[second_cut:])
    kid_B = conditional_replace(B_map, A_excerpt, parent_B[:first_cut]) + A_excerpt + conditional_replace(B_map, A_excerpt, parent_B[second_cut:])    
    
    return kid_A, kid_B


def cx(parent_A, parent_B):
    """
    Crossover function that produces offspring with child bits in same locations as in parents
    """
    length = len(parent_A)
    assert len(parent_A) >= 3
    
    parents = [parent_A, parent_B]
    kids = []
    for parent_idx in range(2):
        parent, other_parent = parents[parent_idx], parents[1 - parent_idx]
        bit = first_bit = parent[0]
        first_round = True
        kid = [bit] + [None] * (length - 1)
        while (bit != first_bit) or first_round:
            first_round = False
            bit_location = other_parent.index(bit)
            bit = parent[bit_location]
            kid[bit_location] = bit
        remaining = [x for x in other_parent if x not in kid]
        for idx, n in enumerate(kid):
            if n is None:
                kid[idx] = remaining.pop(0)
        
        kids.append(kid)
    
    return kids
    

def genetic_step_with_crossover(
    population, 
    crossover_fn, 
    pop_size,
    replication_factor, 
    reproduce_prob, 
    exchange_prob, 
    mutate_prob
):
    """
    Single iteration/step for genetic algorithm with crossover
    """
    new_population = []
    mate_indices = list(np.random.choice(pop_size, size=pop_size, replace=False))
    mates = [population[mate_idx] for mate_idx in mate_indices]
    for idx, sequence in enumerate(population):
        for _ in range(replication_factor):

            if np.random.choice(2, p=[1-reproduce_prob, reproduce_prob]):
                new_population += crossover_fn(sequence, mates[idx])   
            
            if np.random.choice(2, p=[1-mutate_prob, mutate_prob]):
                apply_exchange = np.random.choice(2, p=[1-exchange_prob, exchange_prob])
                new_population.append(exchange(sequence) if apply_exchange else reverse(sequence))

    return new_population


def genetic_crossover(
    graph,
    locations,
    start=None,
    end=None,
    pop_size=20, 
    replication_factor=4, 
    max_iters=200, 
    exchange_prob=0.75, 
    mutate_prob=0.1, 
    reproduce_prob=1.0,
    crossover_fn=pmx,
    initial_state=None,
    max_retries=100,
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
    Genetic algorithm that combines mutations (reverse or exchange) with crossover ('sexual reproduction')
    """
    constraint_fn, cost_fn, constraint_fn_kwargs, cost_fn_kwargs = init_cost_fns(
        constraint_fn, cost_fn, constraint_fn_kwargs, cost_fn_kwargs)

    if encoder is not None:
        if encoder_kwargs is None:
            encoder_kwargs = dict()    
        locations = encoder(locations, **encoder_kwargs)

    if decoder is not None and decoder_kwargs is None:
        decoder_kwargs = dict()

    population = [list(np.random.permutation(locations)) for n in range(pop_size)]
    if initial_state is not None:
        population[0] = initial_state

    for _ in tqdm(range(max_iters)):
        population += genetic_step_with_crossover(
            population, 
            crossover_fn, 
            pop_size, 
            replication_factor, 
            reproduce_prob,
            exchange_prob,
            mutate_prob
        )
        population = selection(
            graph, 
            start, 
            end, 
            population, 
            pop_size, 
            decoder, 
            decoder_kwargs, 
            constraint_fn, 
            constraint_penalty, 
            constraint_fn_kwargs,
            cost_fn, 
            cost_penalty, 
            cost_fn_kwargs
        )

    raw_answer, answer, distance = parse_results(population[0], start, end, graph, decoder, decoder_kwargs)

    if not all(constraint_fn(population[0], **constraint_fn_kwargs)):
        if max_retries == 0:
            return null_result()
        else:
            print(f"Invalid solution found; re-running algorithm ({max_retries} attempts left)")
            raw_answer, answer, distance = genetic_crossover(
                graph, 
                locations, 
                start, 
                end, 
                pop_size, 
                replication_factor, 
                max_iters, 
                exchange_prob,
                mutate_prob,
                reproduce_prob,
                crossover_fn,
                initial_state,
                max_retries - 1,
                encoder,
                encoder_kwargs,
                decoder,
                decoder_kwargs,
                constraint_fn,
                constraint_penalty,
                constraint_fn_kwargs,
                cost_fn, 
                cost_penalty, 
                cost_fn_kwargs,
            )

    return raw_answer, answer, distance
