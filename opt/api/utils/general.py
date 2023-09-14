from opt.visual.visual import plot_graph, animated_plot
from opt.algorithms.naive import naive_tsp
from opt.algorithms.genetic import genetic, genetic_crossover
from opt.algorithms.annealing import anneal, deterministic_anneal, anneal_thermodynamic



def get_single_image_path(image_path, suffix, extension='png'):
    output_path = ""
    if image_path:
        parsed_path = image_path.split(".")
        if len(parsed_path) > 1:
            save_path, ext = parsed_path
        else:
            save_path = parsed_path[0]
            ext = extension
        
        output_path = f"{save_path}_{suffix}." + ext

    return output_path


def format_kwargs(kwargs, save_image_path, display_image, save_animation_path):
    plot_kwargs = format_plot_kwargs(kwargs, save_image_path, display_image)
    anim_kwargs = format_animation_kwargs(kwargs, save_animation_path)
    kwargs = {k: v for k, v in kwargs.items() if (k not in plot_kwargs) and (k not in anim_kwargs) and k != 'baseline_distance'}
    if 'constraint' in kwargs:
        plot_kwargs |= {'constraints_to_plot': kwargs['constraint']}
        anim_kwargs |= {'constraints_to_plot': kwargs['constraint']}

    return plot_kwargs, anim_kwargs, kwargs


def format_plot_kwargs(kwargs, save_image_path, display_image):
    plot_kwargs = {k: v for k, v in kwargs.items() if k in plot_graph.__code__.co_varnames}
    plot_kwargs["save_image_path"] = save_image_path
    plot_kwargs["display_result"] = display_image

    return plot_kwargs


def format_animation_kwargs(kwargs, save_animation_path):
    anim_kwargs = {k: v for k, v in kwargs.items() if k in animated_plot.__code__.co_varnames}
    anim_kwargs["output_path"] = save_animation_path
    return anim_kwargs


def check_algorithm(algorithm_str):
    algorithms = {
        "annealing": anneal,
        "threshold": deterministic_anneal,
        "thermodynamic": anneal_thermodynamic,
        "genetic": genetic,
        "genetic x": genetic_crossover,
        "naive": naive_tsp
    }
    algos = list(algorithms.keys())
    assert algorithm_str in algorithms, f"Please use `{'`, `'.join(algos[:-1])}` or `{algos[-1]}`"


def chunk(all_items, batch_size):
    for i in range(0, len(all_items), batch_size):
        yield all_items[i:i + batch_size]
