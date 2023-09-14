import os
import json
import math
from typing import List, Union

import pandas as pd
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

from opt.utils import conditional_int
from opt.graph import prune_graph, add_distances, add_shortest_paths
from opt.domain import standardize_name, WAREHOUSES, NEW_DCS, ALL_DCS
from opt.io.remote import get_space_locations, get_warehouse_data
from opt.io.gql import get_warehouse_data as get_warehouse_data_legacy, _space_query
from opt.algorithms.reslotting import sd_from_bin_columns, bin_columns_from_sd



# Basic path utilities

def normalize_folder(path: str, char: str='/'):
    return path.rstrip(char) + char

def normalize_fn(fn: str, extension: str=".json"):
    return fn.split(".")[0] + extension

def normalize_path(filename: str, folder_name: str, extension='.json'):
    filename = normalize_fn(filename, extension)
    folder_name = normalize_folder(folder_name)
    path = f"{folder_name}{filename}"

    return path


# JSON

def read_json(filename: str, folder_name: str="."):
    """
    Standard boilerplate code for reading in some JSON
    """
    path = normalize_path(filename, folder_name)
    assert os.path.exists(path), f"{path} could not be found."

    f = open(path, "r")
    return json.load(f)


def write_json(data_dict: dict, filename: str, path="."):
    path = normalize_path(filename, path)
    f = open(path, "w")
    data_str = json.dumps(data_dict)
    f.write(data_str)
    f.close()

    return data_str


# Parse multi-format inputs

def parse_domain_input(domain_id_or_graph: Union[str, nx.classes.graph.Graph]) -> nx.classes.graph.Graph:

    G = domain_id_or_graph
    if type(domain_id_or_graph) == str:
        print(f"Loading graph for {domain_id_or_graph}...")
        domain_name = standardize_name(domain_id_or_graph)
        G = get_domain_graph(domain_name)
        print("Got domain graph")
    
    return G


def within_range(assignment_len, length_range):
    return assignment_len in range(min(length_range), max(length_range))

def bin_in_graph_data(bin, graph_data):
    return bin in graph_data.keys()

def bins_in_graph_data_multi(assignment_data, graph_data):
    return all([bin_in_graph_data(x, graph_data) for x in assignment_data])

def get_assignments_of_len(assignment_data, length_range, randomize=False, num_assignments=None, filter_by_graph=None):
    data = [a for a in assignment_data if 
        (within_range(len(a), length_range) 
            and ((filter_by_graph is None) 
            or (bins_in_graph_data_multi(a, filter_by_graph.bins)))
        )
    ]     
    if randomize:
        data = list(np.random.permutation(data))
    
    return data[:num_assignments]

def parse_assignment_input(
    assignment_path_or_data: Union[str, list, List[list]], 
    random_assignments:bool = False, 
    assignment_length_range: bool = None, 
    num_assignments: bool = None,
    group_by_assignment: bool = True
) -> List[list]:
    """
    Return parsed assignment or pick data (list or list of lists), given either the data itself or its path
    """

    assert not ((random_assignments or assignment_length_range) and (not group_by_assignment)), \
        "Cannot use assignment-sorting args if `group_by_assignment` is False!"

    data = assignment_path_or_data
    if type(assignment_path_or_data) == str:
        print(f"Loading pick data at {assignment_path_or_data}...")
        extension = assignment_path_or_data.split(".")[1]
        if extension == "csv":
            data = bins_from_pick_data(assignment_path_or_data, group_by_assignment=group_by_assignment)
        elif extension == "npy":
            data = np.load(assignment_path_or_data, allow_pickle=True)
        elif extension == "json":
            data = read_json(assignment_path_or_data)

    if random_assignments: 
        data = list(np.random.permutation(data))
    if assignment_length_range:
        get_assignments_of_len(data, assignment_length_range)

    return data[:num_assignments]


def in_bin_column_format(data):
    return not any([x for x in data[0] if x in data[1]])

def parse_reslotting_input(swap_path_or_data, num_swaps=None, return_sources_and_destinations=False):

    data = swap_path_or_data
    if type(swap_path_or_data) == str:
        print(f"Loading swap list at {swap_path_or_data}...")
        extension = swap_path_or_data.split(".")[1]
        if extension == "csv":
            data = load_reslotting_data(swap_path_or_data, num_swaps, return_sd=return_sources_and_destinations)
        elif extension == "npy":
            data = np.load(swap_path_or_data, allow_pickle=True)
        elif extension == "json":
            data = read_json(swap_path_or_data)
    
    if return_sources_and_destinations:
        if in_bin_column_format(data):
            print("Auto-converting data from swaps (Bin A, Bin B) format to (source, destination) format")
            data = sd_from_bin_columns(*data)
    else:
        if not in_bin_column_format(data):
            print("Auto-converting data from (source, destination) format to swaps (Bin A, Bin B) format")
            data = bin_columns_from_sd(*data) 

    data = [data[0][:num_swaps], data[1][:num_swaps]] if type(data) in [tuple, dict, list] else data[:,:num_swaps]
    return data


def parse_capacity_input(capacity_path_or_data, **kwargs):
    data = capacity_path_or_data
    if type(capacity_path_or_data) == str:
        print(f"Loading capacity data at {capacity_path_or_data}...")
        data = load_capacity_data(data)
    
    return data


# Graph I/O

def store_graph(graph: nx.Graph, filename: str="graph_data.json", path: str=".", store_paths=True, store_distances=True, representation="node_link"):
    assert representation in ["adjacency", "node_link", "cytoscape"], "Use `node_link` (default), `adjacency` or `cytoscape` for storage format"
    if representation == "node_link":
        nx_data = json_graph.node_link_data(graph)
    if representation == "adjacency":
        nx_data = json_graph.adjacency_data(graph)
    if representation == "cytoscape":
        nx_data = json_graph.cytoscape_data(graph)

    graph_data = {"format": representation, "nx_data": nx_data}

    if hasattr(graph, "shortest_paths") and store_paths:
        graph_data["shortest_paths"] = graph.shortest_paths

    if hasattr(graph, "distances") and store_distances:
        graph_data["distances"] = graph.distances

    if hasattr(graph, "dimensions"):
        graph_data["dimensions"] = graph.dimensions

    if hasattr(graph, "for_reslotting"):
        graph_data["for_reslotting"] = graph.for_reslotting

    if hasattr(graph, "original_nodes"):
        graph_data["original_nodes"] = graph.original_nodes
    
    if hasattr(graph, "bins"):
        graph_data["bins"] = graph.bins

    path = normalize_folder(path)
    filename = normalize_fn(filename)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    
    data_str = write_json(graph_data, filename, path)

    return data_str


def map_keys_to_int_recursive(to_convert):
    if type(to_convert) == dict:
        return {int(k): map_keys_to_int_recursive(v) for k, v in to_convert.items()}
    else:
        return to_convert


def load_graph(filename: str, path: str=".", include_shortest_paths=True, include_distances=True, include_bins=True):

    path = normalize_path(filename, path)
    assert os.path.exists(path), f"{path} could not be found."

    g_graph = open(path)
    graph_data = json.load(g_graph)
    g_graph.close()

    format = graph_data.get("format")
    if not format: 
        print("Storage format not specified. Trying default (adjacency)")
        format = "node_link"

    nx_data = graph_data.get("nx_data")
    assert nx_data, "Graph lacks networkx-formatted node/edge data!"

    if format == "adjacency":
        G = json_graph.adjacency_graph(nx_data)
    elif format == "node_link":
        G = json_graph.node_link_graph(nx_data)
    if format == "cytoscape":
        G = json_graph.cytoscape_graph(nx_data)

    if "shortest_paths" in graph_data and include_shortest_paths:
        G.shortest_paths = map_keys_to_int_recursive(graph_data["shortest_paths"])

    if "distances" in graph_data and include_distances:
        G.distances = map_keys_to_int_recursive(graph_data["distances"])

    if "dimensions" in graph_data:
        G.dimensions = tuple(graph_data["dimensions"])
    
    if "original_nodes" in graph_data:
        G.original_nodes = graph_data["original_nodes"]
    
    if "bins" in graph_data and include_bins:
        G.bins = graph_data["bins"]

    return G


def load_warehouse_graph(domain_or_name: str, **kwargs):
    """Given a warehouse name ("LA", "KF", etc), SWID, or reverse domain identifier, return its stored graph"""

    name = standardize_name(domain_or_name)
    path = f"data/dcs/nri/{name}/{name}.json"
    assert os.path.exists(path), f"Warehouse data for {domain_or_name} could not be found locally."

    return load_graph(path, **kwargs)


def warehouse_data_from_files(connections_file, output_file):

    f_conn = open(connections_file)
    f_output = open(output_file)

    connections = json.load(f_conn)
    output = json.load(f_output)

    spaces = output["spaces"]

    return connections, spaces


def add_bin_dict_to_graph(graph, domain_ID, use_old_api=False, constraints=None, **kwargs):
    bin_dict = get_bin_dict(domain_ID, use_old_api=use_old_api, constraints=constraints, **kwargs)
    bin_locs = list(bin_dict.values())
    bins = list(bin_dict.keys())

    waypoint_location_dict = nx.get_node_attributes(graph, "pos")
    waypoints, distances = closest_waypoint_multi([bin_locs], waypoint_location_dict)
    waypoints = waypoints[0]
    distances = distances[0]

    bin_info = {
        bin: {
            "pos": bin_locs[idx], 
            "closest_waypoint": waypoints[idx], 
            "distance": distances[idx]
        } for idx, bin in enumerate(bins)
    }
    graph.bins = bin_info

    return graph


def create_bin_info_dict(graph, base_bin_dict: dict) -> dict:
    bin_locs = list(base_bin_dict.values())
    bins = list(base_bin_dict.keys())

    waypoint_location_dict = nx.get_node_attributes(graph, "pos")
    waypoints, distances = closest_waypoint_multi([bin_locs], waypoint_location_dict)
    waypoints = waypoints[0]
    distances = distances[0]

    bin_info = {
        bin: {
            "pos": bin_locs[idx], 
            "closest_waypoint": waypoints[idx], 
            "distance": distances[idx]
        } for idx, bin in enumerate(bins)
    }

    return bin_info
    

def get_bin_info_df(graph_or_domain, download=True):
    dc_name = WAREHOUSES[standardize_name(graph_or_domain)]
    G = parse_domain_input(dc_name, use_cached_graph=not download)
    if hasattr(G, 'bins'):
        space_data_dict = G.bins
    else:
        base_bin_dict = get_space_locations(dc_name)
        space_data_dict = create_bin_info_dict(G, base_bin_dict)
    bin_names = list(space_data_dict.keys())
    field_names = list(space_data_dict[bin_names[0]].keys())
    spaces_df = pd.DataFrame(columns=field_names, index=bin_names)
    for bin in bin_names:
        for field in field_names:
            spaces_df.loc[bin][field] = space_data_dict[bin][field]
    return spaces_df


def from_waypoints(spaces, connections, include_paths=True, include_distances=True, include_bins=True, force_connected=True, use_raw_nodes=False, domain_ID=None, constraints=None, **kwargs):    
    assert not (include_bins and domain_ID is None), "Must specify domain to automatically add relevant spaces to graph."

    G = nx.Graph()
    if not use_raw_nodes:
        original_nodes = []

    for idx, s in enumerate(spaces):
        if use_raw_nodes:
            G.add_node(s, pos = (s["coordinates"]["x"], s["coordinates"]["y"]))
        else:
            G.add_node(idx, pos = (s["coordinates"]["x"], s["coordinates"]["y"]))
            original_nodes.append(s["swid"])

    for v in connections.values():
        if use_raw_nodes:
            G.add_edge(v["_from"], v["_to"], weight=v["distance"])
        else:
            G.add_edge(original_nodes.index(v["_from"]), original_nodes.index(v["_to"]), weight=v["distance"])

    if force_connected:
        G = prune_graph(G)

    nx.set_node_attributes(G, "#AAB4D9", name="node_color")

    if include_paths: add_shortest_paths(G)
    if include_distances: add_distances(G)
    if include_bins: 
        G = add_bin_dict_to_graph(G, domain_ID, use_old_api=(standardize_name(domain_ID) != "LD"), constraints=constraints, **kwargs)

    if not use_raw_nodes:
        G.original_nodes = original_nodes

    return G


def get_bin_dict(domain_ID, base_path="data/dcs/nri", name_key="name", update_local_copy=True, use_old_api=False, force_download=False, constraints=None):
    short_ID = standardize_name(domain_ID)
    base_DC_path = f"{base_path}/{short_ID}"
    write_path = f"{base_DC_path}/{short_ID}_bin_locations.json"

    print(f"Getting space dict for domain {domain_ID}")
    if os.path.exists(write_path):
        if not (force_download or update_local_copy):
            print("Bin location dict already exists.")
            return read_json(write_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)    
    if not os.path.exists(base_DC_path):
        os.makedirs(base_DC_path)

    print(f"Queryng space data for {domain_ID}...")
    if constraints:
        print(f"Using constraint {constraints}")
    if use_old_api:
        space_data = _space_query(domain_ID)
        containers = [space for space in space_data if space[name_key][:3] == f"{short_ID}-"]
        assert containers, f"No spaced found for domain {short_ID}!"
        print("Constructing space data dict")
        space_data_dict = bin_locations_dict(containers)
    else:
        space_data_dict = get_space_locations(domain_ID, constraints=constraints)
    if update_local_copy:
        _ = write_json(space_data_dict, write_path)
    print("Done.")

    return space_data_dict


def save_bin_dicts(overwrite=False, dcs_to_use=ALL_DCS, use_old_api=False, constraints=None):
    for domain_id in dcs_to_use:
        _ = get_bin_dict(WAREHOUSES[domain_id], update_local_copy=overwrite, force_download=overwrite, use_old_api=use_old_api, constraints=constraints)

def stored_locally(name: str):
    """Check whether an NRI warehouse graph with standardized name is stored locally"""
    short_name = standardize_name(name)
    return os.path.exists(f"data/dcs/nri/{short_name}/{short_name}.json")

def get_domain_graph(domain_id, force_download=False, update_local_copy=True, storage_format="node_link", constraints=None, **kwargs):
    """Return a graph representing a warehouse, given its SWID , 2-letter ID, or reverse domain lookup string"""

    # force_download = True
    if not stored_locally(domain_id) or force_download:
        print(f"Loading data for warehouse {domain_id} from server...")
        # data = get_warehouse_data(domain_id)
        short_name = standardize_name(domain_id)
        #constraints = {"z": (2, 10)} if short_name == "LD" else None
        get_fn = get_warehouse_data if short_name == "LD" else get_warehouse_data_legacy
        data = get_fn(domain_id, constraints=constraints)
        assert data, "Error loading warehouse data"

        G = from_waypoints(*data, domain_ID=domain_id, constraints=constraints, **kwargs)
        if update_local_copy:
            print("Storing a local copy...")
            update_warehouse_graph(short_name, G, storage_format, create_ok=True)
            print("Done.")    
    else:
        G = load_warehouse_graph(domain_id, **kwargs)

    return G


def update_warehouse_graph(domain_id, graph, representation="node_link", create_ok=False):
    assert stored_locally(domain_id) or create_ok, f"Graph for {domain_id} not stored; can't perform update!"
    store_graph(graph, f"{standardize_name(domain_id)}", f"data/dcs/nri/{domain_id}", representation=representation)
        

def get_graphs(dcs_to_use=ALL_DCS):
    print(f"Downoading graph data for {', '.join(dcs_to_use).rstrip(',')}")
    doms = [v for k, v in list(WAREHOUSES.items()) if k in dcs_to_use]
    for domain_id in doms:
        print("domainid", domain_id)
        _ = get_domain_graph(domain_id)


# Pick / bin location data

def filter_assignments(pick_data):
    return [assignment for assignment in pick_data if len(assignment) > 2]


def bins_from_pick_data(
    filename: str, 
    folder_name: str=".", 
    location_column="FromLocation", 
    assignment_column="AssignmentNumber", 
    group_by_assignment=True
):
    """
    Extracts a list of pick locations from a .csv file and returns them
    Optionally groups picks by assignment (AssignmentNumber)

    returns: [[Pick locations] per Assignment] if `group_by_assignment` is True, [Pick locations] otherwise
    """
    path = normalize_path(filename, folder_name, ".csv")
    assert os.path.exists(path), f"{path} could not be found."

    df = pd.read_csv(path)
    assert location_column in df.columns, f"Data lacks `{location_column}` column!"

    if group_by_assignment:
        assignments = df[assignment_column].unique()
        return [list(df.loc[df[assignment_column] == assignment_no, location_column]) for assignment_no in assignments]

    return list(df.loc[:,location_column])


def bin_locations_dict(space_data: List[dict], name_field="name", location_field="coordinates", dimensions = ["x", "y"]):
    """Take raw GQL query results for spaces within a domain and return a dict of the form {name: (coords)}"""
    locations_data = dict()
    for space in space_data:
        key = space.get(name_field)
        assert key, "Space lacks a name field!"
        entry = space[location_field]
        coords = []
        for dim in dimensions:
            value = entry.get(dim)
            assert value, f"{dim} coordinate for {key} missing!"
            coords.append(value)
        locations_data[key] = tuple(coords)
    
    return locations_data


def load_reslotting_data(reslotting_data_path, num_swaps=None, return_sd=False):
    data = pd.read_csv(reslotting_data_path).reset_index(drop=True).dropna()
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')].to_numpy()
  
    bin_A_list = [dp[0] for dp in data[:num_swaps]]
    bin_B_list = [dp[1] for dp in data[:num_swaps]]

    if return_sd:
        return sd_from_bin_columns(bin_A_list, bin_B_list)
    else: 
        return (bin_A_list, bin_B_list)


def extract_individual_distances(cumulative_distances_list):
    start = [cumulative_distances_list[0]]
    return start + [cumulative_distances_list[idx] - cumulative_distances_list[idx-1] for idx in range(1, len(cumulative_distances_list))]


def parse_move(move):
    pick_phrase = "Pick item from "
    place_phrase = " in "

    if move[:15] == pick_phrase:
        from_location = item_ref = move.split(pick_phrase)[1]
        to_location = "Cart"
        picker_location = from_location

    elif place_phrase in move:
        first_chunk, to_location = move.split(place_phrase)
        from_location = "Cart"
        item_ref = first_chunk.split("from ")[1]
        picker_location = to_location

    return conditional_int(item_ref), from_location, to_location, picker_location


def name_of(item_ref):
    return f"Items from ({item_ref})"


def format_moves_for_csv(sequence, swap_dict, cart_capacity, batch_position=0):
    """
    Convert a sequence of reslotting 'moves' into [from_location, to_location] format
    """
    sequence_idxs = []
    picker_locations = []
    item_names = []
    from_locations = []
    to_locations = []
    items_in_cart = set()
    items_being_swapped = set()
    items_put_back = set()
    items_in_cart_list = []
    num_items_in_cart = []
    remaining_capacities = []
    swaps_initiated = []
    swaps_completed = []
    total_capacities = [cart_capacity]*len(sequence)

    for idx, move in enumerate(sequence):
        item_ref, from_location, to_location, picker_location = parse_move(move)
        item = name_of(item_ref)
        swap_companion = name_of(swap_dict[item_ref])
        swap = f"{item} <--> {swap_companion}"
        item_names.append(item)
        from_locations.append(from_location)
        to_locations.append(to_location)
        picker_locations.append(picker_location)
        sequence_idxs.append(idx+batch_position+1)

        if to_location == "Cart":
            assert item not in items_in_cart, "Impossible condition: item picked on this move already in cart"
            items_in_cart.add(item)
            if not (item or swap_companion) in items_being_swapped:
                swaps_initiated.append(swap)
                items_being_swapped.add(item)
                items_being_swapped.add(swap_companion)
            else:
                swaps_initiated.append(None)
            swaps_completed.append(None)

        elif from_location == "Cart":
            assert item in items_in_cart, "Impossible condition: item delivered not previously in cart"
            items_in_cart.remove(item)
            items_put_back.add(item)
            if swap_companion in items_put_back:
                swaps_completed.append(swap)
                items_being_swapped.remove(item)
                items_being_swapped.remove(swap_companion)
            else:
                swaps_completed.append(None) 
            swaps_initiated.append(None)

        items_in_cart_list.append(list(items_in_cart) or None)
        used_capacity = len(items_in_cart)
        num_items_in_cart.append(used_capacity)
        remaining_capacities.append(cart_capacity - used_capacity)

    return {
        "Sequence": sequence_idxs,
        "Picker location": picker_locations,
        "Items": item_names,
        "From location": from_locations,
        "To location": to_locations,
        "Swap initiated": swaps_initiated,
        "Swap completed": swaps_completed,
        "Items in cart": items_in_cart_list,
        "Count in cart": num_items_in_cart,
        "Remaining cart capacity": remaining_capacities,
        "Total cart capacity": total_capacities
    }


def swaps_from_sequence(sequence):
    """
    Convert a sequence of reslotting 'moves' to a list of output swaps
    Also does an extra check on the validity of the swap list before saving
    """
    assert len(sequence)%4 == 0, "Input should be a list of A --> B, B --> A transitions"

    pick_phrase = "Pick item from "
    place_phrase = "in "
    warning = "Data in wrong format for swap list (items out of place)"

    bin_A_column = []
    bin_B_column = []

    counter = 0
    item = None

    for move in sequence:
        if counter == 4:
            counter = 0        
        if move[:15] == pick_phrase: 
            last_item = item
            item = move.split(pick_phrase)[1]
            if counter == 0:
                bin_A_column.append(item)
                removed_item = item
            elif counter == 2:
                assert last_item == item
            elif counter == 3:
                assert False, warning
        elif place_phrase in move:
            item = move.split(place_phrase)[1]
            if counter == 1:
                assert removed_item != item
                bin_B_column.append(item)
            elif counter == 2:
                assert False, warning
        counter += 1
    
    return bin_A_column, bin_B_column


def get_swap_distances(distance_list):
    """Extract distances per swap from raw 'move'-based distance data"""
    return [x for idx, x in enumerate(distance_list) if ((idx > 0) and ((idx+1)%4 == 0))]

def save_reslotting_data(
    slot_data,
    swap_dict,
    cart_capacity, 
    output_format, 
    cumulative_distances=True, 
    distance_data=None, 
    save_str=None, 
    data_type="optimized", 
    save_it=True,
    batch_index=None
):
    assert cumulative_distances == (distance_data is not None), "Must supply distance data with `cumulative_distances` flag"
    assert (len(slot_data) == len(distance_data)) or distance_data is None
        
    data_dict = dict()

    if output_format=="move_description":

        data_dict["Moves"] = slot_data
        if cumulative_distances:
            data_dict["Cumulative_distance"] = distance_data
    
    elif output_format=="move_list":
        batch_size = len(slot_data[0]) if batch_index is not None else 0
        batch_index = batch_index or 0
        data_dict = format_moves_for_csv(slot_data, swap_dict, cart_capacity, batch_index*batch_size*2)
        if cumulative_distances:
            data_dict |= {
                "Distance": extract_individual_distances(distance_data), 
                "Cumulative distance": distance_data
            }

    elif output_format=="swaps":
        cart_capacity="1"
        swap_data = swaps_from_sequence(slot_data)
        data_dict["Bin A"] = np.array(swap_data[0])
        data_dict["Bin B"] = np.array(swap_data[1])
        if cumulative_distances:
            distance_data = get_swap_distances(distance_data)
            data_dict["Cumulative_distance"] = distance_data

    df = pd.DataFrame(data_dict)
    if save_it:
        df.to_csv(f"{save_str}_{data_type}_{output_format}_capacity_{cart_capacity}.csv")

    return df


def save_reslotting_data_batch(
    all_slot_data,
    batches,
    cart_capacity, 
    output_format, 
    cumulative_distances=True, 
    all_distance_data=None, 
    save_str=None,
    data_type="optimized", 
    save_it=True
):
    assert cumulative_distances == (all_distance_data is not None), "Must supply distance data with `cumulative_distances` flag"

    batch_dfs = []
    for batch_idx, batch in enumerate(batches):
        batch_size = len(batch[0])
        slot_data = all_slot_data[batch_idx].english
        distance_data = all_distance_data[batch_idx]
        swap_dict = {s: d for s, d in zip(batch[0], batch[1])}

        df = save_reslotting_data(
            slot_data,
            swap_dict,
            cart_capacity,
            output_format,
            cumulative_distances,
            distance_data,
            save_str,
            data_type,
            save_it=False,
            batch_index=batch_idx
        )
        df.insert(1, 'Batch', batch_idx)
        df.insert(2, 'Batch index', range(1, (batch_size*2)+1))
        batch_dfs.append(df)
        
    full_df = pd.concat(batch_dfs, ignore_index=True)

    if save_it:
        batch_size=len(batches[0][0])
        full_df.to_csv(f"{save_str}_{data_type}_{output_format}_capacity_{cart_capacity}_batchsize_{batch_size}.csv")

    return full_df


def save_pick_data(pick_data, cumulative_distances=True, distance_data=None, save_str=None, data_type="optimized", save_it=True):
    assert cumulative_distances == (distance_data is not None), "Must supply distance data with `cumulative_distances` flag"
    if type(distance_data) == list:
        assert (len(pick_data) == len(distance_data)) or distance_data is None

    data_dict = {"Pick_location": pick_data}
    if cumulative_distances:
        data_dict["Cumulative_distance"] = distance_data

    df = pd.DataFrame(data_dict)
    if save_it:
        df.to_csv(f"{save_str}_pick_data_{data_type}.csv")

    return df


def save_pick_data_batch(batches, cumulative_distances=True, all_distance_data=None, save_str=None, data_type="optimized", save_it=True):
    assert cumulative_distances == (all_distance_data is not None), "Must supply distance data with `cumulative_distances` flag"

    batch_dfs = []
    total_num = 0
    for batch_idx, batch in enumerate(batches):
        batch_size = len(batch)
        total_num += batch_size
        distance_data = all_distance_data[batch_idx]

        print("BAT", batch)
        print("DIST", distance_data)
        df = save_pick_data(
            batch,
            cumulative_distances,
            distance_data,
            save_str,
            data_type,
            save_it=False
        )
        df.insert(0, 'Batch', batch_idx)
        df.insert(1, 'Batch index', range(1, (batch_size)+1))
        batch_dfs.append(df)
        
    full_df = pd.concat(batch_dfs, ignore_index=True)
    full_df.insert(0, 'Sequence', range(1, total_num+1))

    if save_it:
        batch_size=len(batches[0][0])
        full_df.to_csv(f"{save_str}_pick_data_{data_type}_batchsize_{batch_size}.csv")

    return full_df


def load_capacity_data(
    path, 
    pickable_col="Pickable", 
    location_type_col="Location Type",
    location_val="Slot",
    address_col="Address", 
    quantity_col="Total Qty"
):
    capacity_df = pd.read_csv(path)
    slots = capacity_df[capacity_df[location_type_col]==location_val]
    pickable_slots = slots[slots[pickable_col]==1]
    totals = pickable_slots.groupby(address_col).sum()[quantity_col]

    return dict(totals)


def get_pick_locations(assignment, bin_location_dict):
    """
    Given an assignment: List[space_ID] 
    and space_location_dict: {space_ID: (x, y)} derived from COSM data, 

    return [(x, y)] for pick locations in assignment"""

    bin_locations = []
    for bin_name in assignment:
        location = bin_location_dict.get(bin_name)
        if not location:
            print(f"Location {bin_name} not found in warehouse space data! Skipping assignment!")
            return None
        bin_locations.append(tuple(location))
    
    return bin_locations


def get_pick_locations_multi(assignments, space_locations):
    skipped = 0
    multi_assignment_data = []
    for assignment in assignments:
        assignment_locations = get_pick_locations(assignment, space_locations)
        if assignment_locations:
            multi_assignment_data.append(assignment_locations)
        else:
            skipped += 1
        
    print(f"Skipped {skipped}/{len(assignments)} assignments!")

    return multi_assignment_data
 

def ingest_pick_data(
    graph, 
    space_locations_fn, 
    pick_locations_fn, 
    space_locations_folder=".", 
    pick_locations_folder=".",
    save_filename=None
):
    """
    Given:
        `graph`: Networkx graph representing an NRI warehouse (or other space)
        `space_locations_fn`: name of .json file containing {bin_name: (x, y)} data for warehouse
        `pick_locations_fn`: name of .csv file containing pick location / assignment data
    (Optional: additional parameters specifying paths to files)
    Return:
        [ [(x, y) for each pick location in an assignment] for each assignment in a list of assignments ]
    """

    space_path = normalize_path(space_locations_fn, space_locations_folder)
    assert os.path.exists(space_path), f"{space_path} could not be found."

    pick_path = normalize_path(pick_locations_fn, pick_locations_folder, '.csv')
    assert os.path.exists(pick_path), f"{pick_path} could not be found."

    print("Loading space locations...")
    space_locs = read_json(space_path)

    print("Done. Loading pick data...")
    pick_locs = bins_from_pick_data(pick_path)

    print("Done. Mapping bin names to locations...")
    pick_coords = get_pick_locations_multi(pick_locs, space_locs)

    print("Done. Building list of closest waypoints...")
    waypoint_locations_dict = nx.get_node_attributes(graph, "pos")
    nodes, distances = closest_waypoint_multi(pick_coords, waypoint_locations_dict)
    if save_filename:
        np.save(f"{save_filename}_nodes", nodes)
        np.save(f"{save_filename}_distances", distances)
    
    return nodes, distances


def distance_euclidean(*input_args):
    """
    Euclidean distance given tuples of (x1, x2), (y1, y2), etc. Where coordinates for a single
    object are e.g. (x1, y1, z1)
    """
    return math.sqrt(sum([math.pow(c1 - c2, 2) for c1, c2 in input_args]))


def closest_waypoint(space_coords: tuple, waypoint_coords_dict: dict, distance_threshold=20.0):
    distance_threshold = 100
    results = []
    distances = []

    for node in waypoint_coords_dict:
        waypoint_coords = waypoint_coords_dict[node]
        dist = distance_euclidean(
            (space_coords[0], waypoint_coords[0]), 
            (space_coords[1], waypoint_coords[1])
        )
        if dist < distance_threshold:
            results.append(node)
            distances.append(dist)

    if len(results) == 0:
        print(f"No suitably close node found for space coordinates {space_coords}!")
        return None, None
    
    closest_idx = distances.index(min(distances))
    return results[closest_idx], distances[closest_idx]


def closest_waypoint_multi(space_coords_list: List[tuple], waypoint_coords_dict: dict, distance_threshold=5.0):
    locations = []
    distances = []

    for space_coords in space_coords_list:
        assignment_locations = []
        assignment_distances = []
        for space_coord in space_coords:
            location, distance = closest_waypoint(space_coord, waypoint_coords_dict, distance_threshold)
            assignment_locations.append(location)
            assignment_distances.append(distance)
        locations.append(assignment_locations)
        distances.append(assignment_distances)
    
    return locations, distances
