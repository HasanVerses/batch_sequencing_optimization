def to_str(data, recursive=False):
    if recursive:
        if type(data) == set:
            return set([to_str(x, recursive=True) for x in data])
        elif type(data) == list:
            return [to_str(x, recursive=True) for x in data]
        elif type(data) == dict:
            return {k: to_str(v, recursive=True) for k, v in data.items()}
        else:
            return str(data)
    else:
        return [str(t) for t in data]   


def conditional_int(string):
    return int(string) if string.isnumeric() else string


def extract_kwargs(fn, kwargs):
    """
    Return only those kwargs that appear in the input variable list of a given function
    """
    if hasattr(fn, 'func'): # Check for functools.partial function
        fn = fn.func 
    return {k: v for k, v in kwargs.items() if k in fn.__code__.co_varnames}


def complement(input: dict, comparison: dict) -> dict:
    if type(input) != type(comparison) != dict:
        raise TypeError("`input` and `comparison` types must be dict!")
    return {k: v for k, v in input.items() if k not in comparison}


def cprint(text: str, condition: bool=True):
    if condition:
        print(text)

def dict_map_recursive(input, mapping_dict):
    """
    Turns a dict map into a function; useful e.g. for partial application in cost function definitions.
    Works recursively on e.g. lists or lists of lists
    """
    if hasattr(input, '__iter__'):
        return [dict_map_recursive(x, mapping_dict) for x in input]
    return mapping_dict[input]
