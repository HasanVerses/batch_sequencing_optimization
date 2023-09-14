import copy
import random
import os
import io
from PIL import Image
import glob

import imageio
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from tqdm import tqdm

from opt.io.local import normalize_folder, normalize_fn
from opt.defaults.cost import convert_to_batch_size_dist



NRI_FORMAT = {"node_labels_on": False, "plot_all_bins": True, "node_size": 30, "scale": 2}
RANDOM_FORMAT = {"node_labels_on": True, "node_size": 120, "scale": 1}
OLD_DEFAULT_FORMAT = {"node_labels_on": True, "node_size": 300, "scale": 1}


SOURCE_COLOR="#fc86ff"
DESTINATION_COLOR="#00b4d9"
PICKED_COLOR="#fc9003"
PICKED_RESLOTTING_COLOR="#ffffff"
TO_RESLOT_COLOR="red"
NODE_COLOR="yellow"
START_COLOR="green"
END_COLOR="#8757b0"
CURRENT_NODE_COLOR="red"
PATH_COLOR="red"
EDGE_COLOR="#aaaaaa"

def check_animated(location_data):
    return "remaining_source" in location_data
 

def process_location_dict(location_data):
    location_data_dict = {"sources": [], "destinations": [], "remaining_sources": [], "remaining_destinations": []}
    animated = False

    if type(location_data) == list:
        location_data_dict["destinations"] = location_data
    elif type(location_data) == dict:
        animated = check_animated(location_data)
        for k, v in location_data.items():
            location_data_dict[k] = v
                
    return location_data_dict, animated


def get_source_color(animated, for_reslotting, sequence):
    if animated:
        if for_reslotting:
            source_col = PICKED_RESLOTTING_COLOR
        else:
            source_col = PICKED_COLOR
    else:
        if sequence:
            source_col = PICKED_COLOR
        else:
            source_col = SOURCE_COLOR
    
    return source_col

            
def set_location_attributes(graph, location_data, use_bins, animated, for_reslotting, sequence):

    if not use_bins:
        [nx.set_node_attributes(graph, {node: DESTINATION_COLOR}, name="node_color") for node in location_data["destinations"]]
        [nx.set_node_attributes(graph, {node: get_source_color(animated, for_reslotting, sequence)}, name="node_color") for node in location_data["sources"]]
        [nx.set_node_attributes(graph, {node: SOURCE_COLOR}, name="node_color") for node in location_data["remaining_sources"]]
            

def get_path_data(graph, path_list, start, end, animated):
    endpoints = [None, None]
    current_loc = None
    path_edges = []
    if path_list:
        if not start:
            start = path_list[0]
        if not end:
            end = path_list[-1]
        if animated:
            current_loc = path_list[-1]
        endpoints = [start, end]
        
        for idx, node in enumerate(path_list):
            if idx < len(path_list)-1:
                path_edges.append((path_list[idx], path_list[idx+1]))
            nx.set_node_attributes(graph, {node: NODE_COLOR}, name="node_color")
        
    return endpoints, path_edges, current_loc


def set_start_end_current_attributes(graph, path, start, end, current_loc, animated):
    if path:
        nx.set_node_attributes(graph, {start: START_COLOR}, name="node_color")
        nx.set_node_attributes(graph, {end: END_COLOR}, name="node_color")
    if animated:
        nx.set_node_attributes(graph, {current_loc: "red"}, name="node_color")


def random_color():
    return f"#{'%06x' % random.randrange(16**6)}"

def get_constraint_colors(num_constraints):
    #TODO: Better quasi-random color generator
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = list(reversed(prop_cycle.by_key()['color']))[:num_constraints]
    while len(colors) < num_constraints:
        colors.append(random_color())

    return colors


def flat_list(list_of_lists):
    return [item for l in list_of_lists for item in l]

def get_plot_data(graph, location_data, use_bins, endpoints, use_edge_weights, node_size, current_loc, for_reslotting, constraints_to_plot=None):

    sources, destinations, remaining_sources, remaining_destinations = location_data.values()

    [_, end] = endpoints

    node_pos = nx.get_node_attributes(graph, 'pos')
    weights = [graph[u][v]['weight'] for u,v in graph.edges()] if use_edge_weights else 1

    node_colors = [n[1]["node_color"] for n in graph.nodes(data=True)]
    constraint_colors = []
    constrained_node_sizes = constrained_node_colors = None
    if constraints_to_plot and not use_bins:
        constraint_colors = get_constraint_colors(len(constraints_to_plot))
        constrained_nodes = flat_list(constraints_to_plot)
        graph_nodes = list(graph)
        node_colors_special = np.zeros(len(graph_nodes), dtype='object')
        for idx, c in enumerate(constraints_to_plot):
            for n in c:
                node_colors_special[graph_nodes.index(n)] = constraint_colors[idx]        
    
        np_nc = np.array(node_colors)
        np_nc[np.nonzero(node_colors_special)] = node_colors_special[np.nonzero(node_colors_special)]
        constrained_node_colors = list(np_nc)
        constrained_node_sizes = [node_size*0.5 if n in constrained_nodes else 0 for n in graph.nodes()]

    node_sizes = [
        node_size*1.5 if ((n in destinations or n in remaining_destinations) and not use_bins)
        else node_size*1.5 if ((n in sources or n in remaining_sources) and not use_bins)
        else node_size*3 if n in endpoints
        else node_size*0.5
        for n in graph.nodes()
    ]
    destination_sizes = remaining_d_sizes = remaining_source_sizes = None

    current_loc_sizes = [node_size*1.5 if n == current_loc else 0 for n in graph.nodes()]
    end_sizes = [node_size*3 if n in endpoints else 0 for n in graph.nodes()]
    end_cols = [END_COLOR if n == end else START_COLOR for n in graph.nodes()]

    destination_sizes = [node_size*2 if n in destinations else 0 for n in graph.nodes()]

    if for_reslotting:
        remaining_d_sizes = [node_size*2 if n in remaining_destinations else 0 for n in graph.nodes()]
        remaining_source_sizes = [node_size if n in remaining_sources else 0 for n in graph.nodes()]

    return (
        node_pos, 
        node_colors, 
        node_sizes, 
        weights, 
        end_cols, 
        end_sizes, 
        current_loc_sizes, 
        destination_sizes, 
        remaining_d_sizes, 
        remaining_source_sizes, 
        constraint_colors,
        constrained_node_colors,
        constrained_node_sizes
    )

def add_marker(patches, color, label, marker_type='o', markersize=8):
    marker = mlines.Line2D([], [], color=color, marker=marker_type, linestyle='None', markersize=markersize, label=label)
    patches.append(marker)

    return patches


def add_legend(ax, path, location_data, animated, for_reslotting, sequence, use_bins, pick_labels_on, constraint_colors):

    patches = []
    if path is not None: 
        patches = add_marker(patches, START_COLOR, 'start node')
        patches = add_marker(patches, END_COLOR, 'end node')
        patches = add_marker(patches, NODE_COLOR, 'node on path')
    
    if location_data is not None:
        sources, destinations, remaining_sources, remaining_destinations = location_data.values()

        if remaining_sources or animated:
            patches = add_marker(patches, SOURCE_COLOR, 'source')

        if sources:
            if not for_reslotting:
                if animated:
                    patches = add_marker(patches, PICKED_COLOR, 'picked')
                else:
                    if sequence:
                        patches = add_marker(patches, PICKED_COLOR, 'source')
                    else:
                        patches = add_marker(patches, SOURCE_COLOR, 'source')
            else:
                if not animated:
                    if not use_bins:
                        patches = add_marker(patches, PICKED_COLOR, 'source')
                    if pick_labels_on:
                        patches = add_marker(patches, SOURCE_COLOR, 'source')
        
        if destinations or remaining_destinations:
            patches = add_marker(patches, DESTINATION_COLOR, 'destination')
        
        if for_reslotting:
            patches = add_marker(patches, DESTINATION_COLOR, 'reslotted', marker_type='x')

        if animated:
            patches = add_marker(patches, CURRENT_NODE_COLOR, 'current location')
        
        for idx, c in enumerate(constraint_colors):
            patches = add_marker(patches, c, f'constraint group {chr(65+idx)}')
        
    if patches:
        ax.legend(bbox_to_anchor=(1.05, 1), handles=patches)


def plot_bins(graph, ax, bin_data, bin_sequence, waypoint_sequence, plot_all_bins, plot_all_bin_labels, plot_bin_labels, node_pos, pick_labels_on, for_reslotting):

    picked, reslotted, remaining_source, remaining_destination = bin_data.values()
    bins_decoded = bin_sequence.decoded if bin_sequence else None

    if plot_all_bins or bin_data:
        assert hasattr(graph, "bins"), "Graph lacks bin location data!"

        if plot_all_bins:
            all_x, all_y = [], []            
        if picked:
            picked_x, picked_y = dict(), dict()
        if reslotted:
            reslotted_x, reslotted_y = dict(), dict()        
        if remaining_source:
            source_x, source_y = dict(), dict()
        if remaining_destination:
            destination_x, destination_y = dict(), dict()
        if bin_sequence:
            sequence_x, sequence_y = dict(), dict()

        for bin in graph.bins:
            coords = graph.bins[bin]["pos"]

            if plot_all_bins:
                all_x.append(coords[0])
                all_y.append(coords[1])
            
            if picked and bin in picked:
                picked_x[bin] = coords[0]
                picked_y[bin] = coords[1]
                if plot_bin_labels:
                    ax.annotate(bin, coords[:2])

            if reslotted and bin in reslotted:
                reslotted_x[bin] = coords[0]
                reslotted_y[bin] = coords[1]
                if plot_bin_labels:
                    ax.annotate(bin, coords[:2])

            if remaining_source and bin in remaining_source:
                source_x[bin] = coords[0]
                source_y[bin] = coords[1]
                if plot_bin_labels:
                    ax.annotate(bin, coords[:2])

            if remaining_destination and bin in remaining_destination:                
                destination_x[bin] = coords[0]
                destination_y[bin] = coords[1]
                if plot_bin_labels:
                    ax.annotate(bin, coords)

            if bin_sequence and bin in bins_decoded:
                sequence_x[bin] = coords[0]
                sequence_y[bin] = coords[1]

        if plot_all_bins:
            ax.scatter(all_x, all_y)
            if plot_all_bin_labels:
                bin_names = list(graph.bins.keys())
                for idx, coord in enumerate(zip(all_x,all_y)):
                    text = bin_names[idx]
                    ax.annotate(text, coord)
        
        if reslotted:
            x, y = [reslotted_x[bin] for bin in reslotted], [reslotted_y[bin] for bin in reslotted]
            ax.scatter(x, y, color=DESTINATION_COLOR, marker='x', label='reslotted')
        
        if picked and not for_reslotting:
            x, y = [picked_x[bin] for bin in picked], [picked_y[bin] for bin in picked]
            ax.scatter(x, y, color=PICKED_COLOR, marker='o', facecolor='none')

        if remaining_source:
            x, y = [source_x[bin] for bin in remaining_source], [source_y[bin] for bin in remaining_source]
            ax.scatter(x, y, color=SOURCE_COLOR, label='remaining sources')

        if remaining_destination:
            x, y = [destination_x[bin] for bin in remaining_destination], [destination_y[bin] for bin in remaining_destination]
            ax.scatter(x, y, color=DESTINATION_COLOR, facecolors='none', s=50, label='remaining destinations')
        
        if bin_sequence:
            bins_x, bins_y = [sequence_x[bin] for bin in bins_decoded], [sequence_y[bin] for bin in bins_decoded]
            bin_types = [bin_sequence[idx]['type'] for idx in range(len(bin_sequence))]

        if waypoint_sequence is not None:
            for idx, waypoint in enumerate(waypoint_sequence):
                destination_coords = node_pos[waypoint]
                ax.plot([bins_x[idx], destination_coords[0]], [bins_y[idx], destination_coords[1]], color='black')
                if bin_sequence and pick_labels_on:
                    bin_type = bin_types[idx]
                    ax.annotate(
                        idx+1, 
                        (bins_x[idx] + 1 + 3*(bin_type == 'destination'), bins_y[idx]), 
                        color=(DESTINATION_COLOR if bin_types[idx] == 'destination' else SOURCE_COLOR), 
                        fontweight='bold'
                    )


def plot_pick_labels(ax, sequence, waypoint_sequence, node_pos):
    node_types = [sequence[idx]['type'] for idx in range(len(sequence))]

    for idx, waypoint in enumerate(waypoint_sequence):
        node_type = node_types[idx]
        node_coords = node_pos[waypoint]
        ax.annotate(
            idx+1, 
            (node_coords[0] + (0.2 - 0.4*(node_type == 'destination')), node_coords[1] + 0.2), 
            color=(DESTINATION_COLOR if node_types[idx] == 'destination' else SOURCE_COLOR), 
            fontweight='bold'
        )


def distance_annotations(ax, sequence, plot_distance, plot_baseline):
    if (sequence is not None) and (plot_distance or plot_baseline):

        x_margin = 0.02
        text_height = 0.5

        texts = []
        if plot_distance:
            texts.append(f'Distance: {sequence.distance}')
        if plot_baseline and hasattr(sequence, 'baseline'): # NOTE: This silently fails if sequence has no "baseline" attribute
            texts.append(f'Baseline: {sequence.baseline}')
            if plot_distance:
                texts = [f'Difference: {sequence.baseline - sequence.distance}'] + texts
        
        for idx, text in enumerate(texts):
            if (len(texts) > 1 and idx == 1) or len(texts) == 1: # Hack: this should be the optimized distance
                ax.text(x_margin, text_height*idx, text, style='oblique')
            else:
                ax.text(x_margin, text_height*idx, text)


def format_plot(x_scale, y_scale, scale, cart_capacity, plot_distance, plot_baseline):
    if scale and (x_scale or y_scale):
        print("Warning: `x_scale` and `y_scale` override passed `scale` parameter.")
    x_scale = x_scale or scale
    y_scale = y_scale or scale

    base_xdim = 6
    base_ydim = 4
    if cart_capacity:
        base_xdim += 2
    if plot_distance or plot_baseline:
        base_ydim += 0.4

    width = base_xdim*x_scale
    height = base_ydim*y_scale

    sidebar_ax = info_ax = None

    if cart_capacity:
        _, axd = plt.subplot_mosaic(
            [['main', 'capacity'],
             ['info', 'capacity']],
             gridspec_kw={'width_ratios': [6, 2], 'height_ratios': [10, 1]},
             figsize=(width, height), constrained_layout=True
        ) 
        main_ax = axd['main']
        sidebar_ax = axd['capacity']

    else:
        _, axd = plt.subplot_mosaic(
            [['main'],
             ['info']],
             gridspec_kw={'height_ratios': [10, 1]},
             figsize=(width, height), constrained_layout=True
        ) 
        main_ax = axd['main']
        info_ax = axd['info']

    info_ax = axd['info']
    info_ax.set_axis_off()

    if cart_capacity:
        sidebar_ax.annotate(
            "Cart",
            (0.1, 0.42)
        )
        cart_slots = [Rectangle((0.1, 0.0 + i*(0.4/cart_capacity)), 0.8, 0.4/cart_capacity) for i in range(cart_capacity)]
        pc = PatchCollection(cart_slots, facecolor='#cccccc', edgecolor='#ffffff')
        sidebar_ax.add_collection(pc)
        sidebar_ax.axis('off')

    return main_ax, sidebar_ax, info_ax


def plot_graph(
    graph,
    sequence=None,
    sources=None,
    destinations=None,
    path=None,
    start=None,
    end=None,
    animation_idx=None,
    edge_labels_on=False, 
    node_labels_on=False,
    pick_labels_on=True,
    node_size=30, 
    path_width=3.5,
    scale=2,
    x_scale=None,
    y_scale=None,
    use_edge_weights=False,
    display_result=True,
    save_image_path="",
    plot_all_bins=False,
    plot_all_bin_labels=False,
    plot_bin_labels=False,
    plot_distance=True,
    plot_baseline=True,
    constraints_to_plot=None
):
    """
    Use networkx and matplotlib to plot a spatial graph and locations and routes defined on it.

    Parameters:

        `graph`: Networkx Graph object: Graph to plot
        `sequence`: Sequence object to plot; contains sources, destinations, and optionally animation data
        `sources`: A list of sources to plot (overrides source data provided via `sequence`)
        `destinations`: A list of sources to plot (overrides destination data provided via `sequence`)
        `path`: List of node identifiers in path (overrides path data provided via `sequence`)
        `start`: Start node identifier (if not specified, defaults to first node in `path`)
        `end`: End/goal node identifier (if not specified, defaults to last node in `path`)
        `animation_idx`: Int specifying which frame of animation we're on (for saving; doubles as "animated" flag)
        `edge_labels_on`: bool: Toggles printing edge labels
        `node_labels_on`: bool: Toggles printing node labels
        `pick_labels_on`: bool: Toggles whether to annotate nodes/bins with sequence #s
        `node_size`: int: Visual size of nodes in plot
        `path_width`: int: Visual width of path edges in plot
        `scale`: float: Scale of plot
        `x_scale`, `y_scale`: Independently adjust scale along x and y dimensions (overrides `scale`)
        `use_edge_weights`: bool: Toggles using edge weights to set visual edge widths
        `display_result`: bool: Toggles whether to display the plot onscreen
        `save_image_path`: str: Path at which to save plot to disk; if not supplies, no plot will be saved
        `plot_all_bins`: bool: If True, plot locations of all bins in a warehouse.
        `plot_all_bin_labels`: bool: If True, label all bin locations with bin identifier.
        `plot_bin_labels`: bool: If True, label all bin locations in pick/reslot sequence with bin identifier.
        `constraints_to_plot`: List[set]: Constraints on sequence to visualize along with data.
    """
    G_copy = copy.deepcopy(graph)
    G_edgeless = nx.create_empty_copy(G_copy, with_data=True)
    animated = animation_idx is not None
    cart_capacity = sequence.cart_capacity if sequence else None
    plot_distance = plot_distance if sequence else False
    plot_baseline = plot_baseline if sequence else False
    if not animated:
        cart_capacity=None

    ax, sidebar_ax, info_ax = format_plot(x_scale, y_scale, scale, cart_capacity, plot_distance, plot_baseline)
    for_reslotting = sequence.for_reslotting if sequence else False
    location_data = sequence.path_slice(animation_idx) if sequence is not None else \
        {"sources": [], "destinations": [], "remaining_sources": [], "remaining_destinations": []}
    
    if sources:
        location_data['sources'] = sources
    if destinations:
        location_data['destinations'] = destinations
    # TODO: Un-hack this
    use_bins = False
    if (sequence and sequence.use_bins) or plot_all_bins:
        use_bins = True

    if path is None:
        path = sequence.path if sequence is not None else None
    waypoints, _ = sequence.waypoints_slice(animation_idx) if sequence else (None, None)
    end = end or (sequence.end if sequence else None)
    
    # Get properties of path; also set path node colors
    [start, end], path_edges, current_loc = get_path_data(G_copy, path, start, end, animated)
    # Set node attributes for locations
    set_location_attributes(G_copy, location_data, use_bins, animated, for_reslotting, sequence)
    # Set node attributes for special (start, end, location) points (overwrites previous settings)
    set_start_end_current_attributes(G_copy, path, start, end, current_loc, animated)

    # Get dicts for node properties; also define node sizes based on properties
    ( node_pos, 
    node_colors, 
    node_sizes, 
    weights, 
    end_cols, 
    end_sizes, 
    current_loc_sizes, 
    destination_sizes, 
    r_destination_sizes, 
    r_source_sizes, 
    constraint_colors,
    constrained_node_colors,
    constrained_node_sizes ) = get_plot_data(
        G_copy,
        location_data,
        use_bins,
        [start, end], 
        use_edge_weights, 
        node_size, 
        current_loc,
        for_reslotting,
        constraints_to_plot
    )

    if use_bins:
        plot_bins(
            G_copy, 
            ax,
            location_data, 
            sequence, 
            waypoints, 
            plot_all_bins, 
            plot_all_bin_labels, 
            plot_bin_labels, 
            node_pos,
            pick_labels_on,
            for_reslotting
        )
    else:
        if sequence and pick_labels_on:
            plot_pick_labels(ax, sequence, waypoints, node_pos)

    # Draw graph
    nx.draw(
        G_copy, 
        ax=ax,
        with_labels=node_labels_on, 
        node_size=node_sizes, 
        pos=node_pos,
        edge_color=EDGE_COLOR,
        node_color=node_colors, 
        width=weights
    )
    # Due to odd behavior of `nx.draw`, re-draw nodes that should be on top (TODO: improve efficiency)
    if end_sizes is not None:
        nx.draw(G_edgeless, ax=ax, pos=node_pos, node_size=end_sizes, node_color=end_cols)
    if r_destination_sizes is not None:
        nx.draw(G_edgeless, ax=ax, pos=node_pos, node_size=r_destination_sizes, node_color=DESTINATION_COLOR)
    if r_source_sizes is not None:
        nx.draw(G_edgeless, ax=ax, pos=node_pos, node_size=r_source_sizes, node_color=SOURCE_COLOR)
    if destination_sizes is not None and location_data['sources']:
        nx.draw(G_edgeless, ax=ax, pos=node_pos, node_size=destination_sizes, node_color=DESTINATION_COLOR, node_shape="x")
    if constraint_colors:
        nx.draw(G_edgeless, ax=ax, pos=node_pos, node_size=constrained_node_sizes, node_color=constrained_node_colors, node_shape="s")
    if current_loc_sizes is not None:
        nx.draw(G_edgeless, ax=ax, pos=node_pos, node_size=current_loc_sizes, node_color=CURRENT_NODE_COLOR)

    if edge_labels_on:
        nx.draw_networkx_edge_labels(G_copy, node_pos, ax=ax)
    if path is not None:
        nx.draw_networkx_edges(G_copy, ax=ax, pos=node_pos, edgelist=path_edges, width=path_width, edge_color='r')

    # Add cart contents visualization (if animated)
    if cart_capacity:
        slot_height = 0.4/cart_capacity
        text_height = 0.05
        left_margin = 0.01
        cart_idx = 0
        for loc in location_data['sources']:
            if sequence.swap_dict[loc] not in location_data['destinations']:
                sidebar_ax.annotate(f'Items from {loc}', (0.1 + left_margin, 0.4 - text_height - cart_idx*slot_height))
                cart_idx += 1

    distance_annotations(info_ax, sequence, plot_distance, plot_baseline)
    add_legend(ax, path, location_data, animated, for_reslotting, sequence, use_bins, pick_labels_on, constraint_colors)
    
    if save_image_path:
        plt.savefig(save_image_path, bbox_inches='tight')

    if display_result:
        plt.show()

    im = None
    if animated:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        im = Image.open(buf)

    plt.clf()
    plt.close("all")

    if im:
        return im


def process_io_args(output_path, frame_folder_name, delete_frames):
    split_path = output_path.split("/")
    contains_folder = len(split_path) > 1
    if frame_folder_name:
        assert not os.path.exists(frame_folder_name), f"Folder '{frame_folder_name}' already exists!"
        os.makedirs(frame_folder_name)
    output_path = normalize_fn(output_path, extension=".gif") if output_path else "tsp_plot.gif"
    frame_name = output_path.split(".")[0] if not contains_folder else split_path[-1].split(".")[0]

    assert frame_folder_name or not delete_frames, "Please supply a folder in which to store temporary .png files!"
    frame_folder_name = normalize_folder(frame_folder_name) if frame_folder_name else ""

    return output_path, frame_name, frame_folder_name


def animated_plot(
    graph, 
    sequence,
    output_path=None,
    frame_folder_name="tsp_plot_frames",
    save_gif=True,
    start_delay=10,
    end_delay=10,
    pick_delay=2,
    max_frames=None,
    delete_frames=True,
    fps=72,
    node_labels_on=False,
    **kwargs
):
    """
    Create an animatied plot of a task on a graph.

    Parameters:

    ` `graph` : Augmented nx.Graph on which data will be plotted
      `sequence`: sequence.Sequence instance containing locations (bins or waypoints) to visit on the graph, 
        together with their types (`source` or `destination`; used for reslotting), cart capacity, etc.
      `output_path`: Path (including file extension) to saved animation
      `frame_folder_name`: Where to store frames for animation
      `save_gif`: bool: Toggles whether to save a GIF (or i.e. just the individual animation frames as .pngs)
      `start_delay`: Delay in the animation before beginning sequence
      `end_delay`: Delay in the animation after finishing sequence
      `pick_delay`: Delay in the animation while picking/reslotting item
      `max_frames`: Maximum number of frames to use for output animation
      `delete_frames`: Bool indicating whether to delete frames once GIF is constructed
      `fps`: frames per second of animation

      Additional kwargs are passed on to `plot_graph`.
    """
    output_path, frame_name, frame_folder_name = process_io_args(output_path, frame_folder_name, delete_frames)

    kwargs["node_labels_on"] = node_labels_on

    for idx in tqdm(range(len(sequence.path))):
        im = plot_graph(
            graph, 
            sequence=sequence,
            path = sequence.path[:idx+1],
            animation_idx=idx,
            display_result=False,
            **kwargs
        )

        repeats = 1
        if idx == 0:
            repeats = start_delay
        elif idx == len(sequence.path) - 1:
            repeats = end_delay
        elif idx in sequence.indices[1:-1]:
            repeats = pick_delay

        for r in range(repeats):
            save_image_path = f"{frame_folder_name}{frame_name}_{idx:04d}{r:02d}.png"
            im.save(save_image_path)

    if save_gif:
        make_gif(frame_folder_name, output_path, max_frames=max_frames, fps=fps)
        if delete_frames:
            filenames = glob.glob(frame_folder_name + "/*.png")
            for fn in filenames:
                try:
                    os.remove(fn)
                except:
                    print(f"Could not remove {fn}")
            try:
                os.rmdir(frame_folder_name)
            except:
                print(f"Could not remove folder {frame_folder_name}!")
                    

def make_gif(frames_folder_name, output_fn = "tsp_plot.gif", max_frames=None, fps=100):
    images = []
    fns = sorted(glob.glob(f"{frames_folder_name}/*.png"))[:max_frames]
    for fn in fns:
        images.append(imageio.imread(fn, pilmode='RGB'))
    print("Saving GIF at", output_fn)
    imageio.mimsave(output_fn, images, format = 'GIF-PIL', fps = fps)


def add_labels(plot, x,y, upper=False):
    for i in range(len(x)):
        e = 13.3
        if not upper:
            e = np.Inf
        plot.text(i, min(e,int(y[i]) + 1.1), int(y[i]), ha = 'center')  


def plot_energy_data(energy_data, ylabel="Energy", save_path='./energies.png'):
    plt.plot(range(len(energy_data)), energy_data)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.clf()
    return


def live_plot_annealing_data(Ts=None, Es=None):
    assert Ts or Es is not None, "Nothing to plot!"
    plt.clf()
    if Ts:
        plt.plot(range(len(Ts)), Ts, label='temperature')
    if Es:
        plt.plot(range(len(Es)), Es, label='energy')
    plt.draw()
    plt.pause(0.00001)

def plot_graph_with_cluster_ids(G, cluster_assignments, vary_sizes=False, color_skew_factor=1, save_path=None, display=True):

    pos =nx.get_node_attributes(G, "pos")
    N = G.number_of_nodes()

    cluster_colors = np.zeros(N)

    node_sizes = [3] * N
    for c_i, cluster_ids in enumerate(cluster_assignments):
        for id_i in cluster_ids:
            cluster_colors[id_i] = c_i*color_skew_factor
            if vary_sizes:
                node_sizes[id_i] = 50

    nx.draw(G, pos = pos, node_color = cluster_colors, node_size=node_sizes)
    if save_path is not None:
        plt.savefig(save_path)
    if display:
        plt.show()
    
    plt.clf()
    return

def plot_overlaid_assignment_histograms(reference_dist, assignment_vec_to_compare, bin_axis_offset = 0.2, bar_width = 0.4):
    """ 
    Function for visualizing a reference distribution `reference_dist` with keys being values of the bin-centers (x-axis) and values
    being the probability for the corresponding bin (y-axis) to an optimized vector of assignments, which is a list of lists,
    where each sub-list `assignment_vec_to_compare[i]` contains the assignments to cluster `i`. The vector of clusters is converted to a size 
    histogram which is then overlaid as a bar chart on the reference distribution
    """

    ref_dist = np.array(list(reference_dist.items()))
    bin_values, ref_probs = ref_dist[:,0], ref_dist[:,1] # get the bins and the probabilities of the reference distribution

    # convert list of cluster assignments to a probability histogram of cluster sizes, digitized using the same bins as the reference size distribution
    p_probs = convert_to_batch_size_dist(assignment_vec_to_compare, bin_values) # convert current cluster size distribution to batch size distribution

    fig, ax = plt.subplots(figsize = (10,8))
    # ax.plot(bin_values, ref_probs, label = 'Desired assignment distribution', lw = 3.0)
    # ax.plot(bin_values, p_probs, label = 'Optimized assignment distribution', lw = 3.0)

    ax.bar(bin_values - bin_axis_offset, ref_probs, bar_width, label = 'Desired assignment distribution')
    ax.bar(bin_values + bin_axis_offset, p_probs, bar_width, label = 'Optimized assignment distribution')
    plt.show()

    plt.clf()
    return

def plot_overlaid_assignment_vectors(ref_vec, optimized_vec, bin_axis_offset=0.2, bar_width=0.4, save_path=None, display=True, sort=False):

    if sort:
        ref_vec = sorted(ref_vec)
        optimized_vec = sorted(optimized_vec)
    num_assignments = range(len(ref_vec))
    assert num_assignments == range(len(optimized_vec)), "Reference and target vectors must be same length!"
    num_assignments = np.array(num_assignments)
    plt.bar(num_assignments - bin_axis_offset, ref_vec, bar_width, label="Reference assignment sizes vector")
    plt.bar(num_assignments + bin_axis_offset, optimized_vec, bar_width, label="Assignment sizes output by clustering")
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    if display:
        plt.show()

    plt.clf()
    return 
