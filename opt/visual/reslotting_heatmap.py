import io
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from PIL import Image
from tqdm import tqdm

from opt.api.snaking import parse_domain_input
from opt.domain import parse_bin_label



# Warehouse - Aisle - Section (of aisle) - Level (height) - Bins/columns (A-G/I)
EDGE_COLOR="#aaaaaa"


def format_plot(x_scale, y_scale, scale):
    if scale and (x_scale or y_scale):
        print("Warning: `x_scale` and `y_scale` override passed `scale` parameter.")
    x_scale = x_scale or scale
    y_scale = y_scale or scale

    width = 6*x_scale
    height = 4*y_scale

    # plt.figure(figsize=(width, height))
    fig, ax = plt.subplots(1, 1)
    fig.set_figwidth(width)
    fig.set_figheight(height)

    return fig, ax


def get_plot_data(graph, use_edge_weights, node_size, edges_on, nodes_on):

    node_pos = nx.get_node_attributes(graph, 'pos')
    weights = [graph[u][v]['weight']*edges_on for u,v in graph.edges()] if use_edge_weights else 1*edges_on
    node_colors = [n[1]["node_color"] for n in graph.nodes(data=True)]

    node_sizes = node_size*nodes_on

    return node_pos, node_colors, node_sizes, weights


# def add_marker(patches, color, label, marker_type='o', markersize=8):
#     marker = mlines.Line2D([], [], color=color, marker=marker_type, linestyle='None', markersize=markersize, label=label)
#     patches.append(marker)

#     return patches


def add_marker(patches, facecolor, edgecolor, label, alpha=1, marker_type='o', markersize=8):
    marker = Rectangle(
                    (0, 0),
                    2,
                    1, 
                    alpha=alpha,
                    facecolor=facecolor, 
                    edgecolor=edgecolor,
                    label=label
                )
    patches.append(marker)

    return patches


def add_legend(bin_color, max_percent):

    patches = []
    patches = add_marker(patches, bin_color, bin_color, f"Bin column ({max_percent:.2f}% of tasks)")
    patches = add_marker(patches, bin_color, bin_color, f"Bin column ({max_percent/2:.2f}% of tasks)", alpha=0.5)
    patches = add_marker(patches, 'none', '#000000', "Section")
    patches = add_marker(patches, 'none', '#f57242', "Aisle")

    l = plt.legend(handles=patches, loc='upper right', prop={'size': 20})
    for idx, t in enumerate(l.get_texts()):
        if idx < 2:
            t.set_color('red')
        if idx == 3:
            t.set_color('#f57242')


def filter_unchanged_df_rows(df, state1_col, state2_col):
    """Returns rows of df which has different values for col1 and col2."""
    return df[df[state1_col] != df[state2_col]].reset_index(drop=True)


def parse_pick_data(df, location_col, post_loc_col, task_col):

    if post_loc_col:
        df = filter_unchanged_df_rows(df, location_col, post_loc_col)

    total_tasks = df[task_col].sum()
    tasks_per_bin= df.groupby(by=[location_col])[task_col].sum()
    tasks = dict(tasks_per_bin)
    percents = tasks_per_bin.div(total_tasks)

    max_popularity = max(tasks_per_bin)
    alphas = dict(tasks_per_bin.div(max_popularity))

    return tasks, total_tasks


def build_dc_dict(graph, tasks): #alphas_dict, bin_width=4, bin_height=8):

    aisles = dict()
    most_popular = 0

    for bin in tqdm(graph.bins):
        bin_data = parse_bin_label(bin, reject_nonstandard=True)
        if not bin_data[-1].isalpha():
            continue
        domain, aisle, section, _, column = bin_data

        task_count = tasks.get(bin) or 0.

        entry = {'alpha': task_count, 'coords': graph.bins[bin]['pos']}
        if aisle in aisles:
            if section in aisles[aisle]:
                if column in aisles[aisle][section]:
                    aisles[aisle][section][column]['alpha'] += task_count
                else:
                    aisles[aisle][section][column] = entry
            else:
                aisles[aisle][section] = {column: entry}
        else:
            aisles[aisle] = {section: {column: entry}}

        most_popular = max(most_popular, aisles[aisle][section][column]['alpha'])

    return aisles, domain, most_popular


def draw_rectangles(dc_dict, ax, domain, most_popular, bin_color, space_labels_on, bin_width=0.3, bin_height=0.4):
    rectangles = []

    x_margin = 0.2
    y_bottom_margin = 0.1
    y_top_margin = 0.2
    sec_label_offset = 0.3
    aisle_label_offset = 0.6
    aisle_label_left_offset = -1

    all_xs, all_ys = [], []
    for aisle in dc_dict:
        aisle_dict = dc_dict[aisle]
        xs, ys = [], []
        for section in aisle_dict:
            section_dict = aisle_dict[section]
            sxs, sys = [], []
            for column in section_dict:
                column_dict = section_dict[column]
                cxs, cys = [], []

                (x, y) = column_dict['coords']
                rectangles.append(
                    Rectangle((x, y), bin_width, bin_height, alpha=column_dict['alpha']/most_popular, facecolor=bin_color, edgecolor=bin_color)
                )
                all_xs.append(x)
                all_ys.append(y)
                xs.append(x)
                ys.append(y)
                sxs.append(x)
                sys.append(y)
                cxs.append(x)
                cys.append(y)
                #if space_labels_on:
                    #ax.annotate(row, (x, y), c='green')
                column_x, column_y = min(cxs), min(cys)
                if space_labels_on:
                    ax.annotate(column, (column_x, column_y + y_bottom_margin), fontsize='xx-small', c='red')
            section_x, section_y = min(sxs), min(sys)
            rectangles.append(
                Rectangle(
                    (section_x - x_margin, section_y - y_bottom_margin), 
                    (max(sxs) + x_margin + bin_width) - section_x, 
                    (max(sys) + y_top_margin + bin_height) - section_y, 
                    facecolor='none', 
                    edgecolor="#000000"
                )
            )
            if space_labels_on:
                ax.annotate(section, (section_x, section_y + sec_label_offset), fontsize='x-small')
        aisle_x, aisle_y = min(xs), min(ys)
        rectangles.append(
            Rectangle(
                (aisle_x - x_margin, aisle_y - y_bottom_margin), 
                (max(xs) + x_margin + bin_width) - aisle_x, 
                (max(ys) + y_top_margin + bin_height) - aisle_y, 
                facecolor='none', 
                edgecolor='#f57242'   #"#fcba03"
            )
        )
        if space_labels_on:
            ax.annotate(aisle, (aisle_x + aisle_label_left_offset, aisle_y), fontsize='small', c='#f57242')

    # rectangles.append(
    #     Rectangle(
    #         (min(all_xs) - 4, min(all_ys) - 4), 
    #         (max(all_xs) + 4 + bin_width) - min(all_xs), 
    #         (max(all_ys) + 4 + bin_height) - min(all_ys), 
    #         facecolor='none', 
    #         edgecolor='#000000',
    #         linewidth=4.
    #     )
    # )
    # if space_labels_on:
    #     ax.annotate(domain, (min(all_xs) - 4, max(all_ys) - 4), fontsize='x-large')

    ax.add_collection(PatchCollection(rectangles, match_original=True))
    return ax
    

def heatmap_viz(
    domain_id_or_graph,
    reslot_algo_output,
    location_col='Location',
    post_loc_col='final_loc_2',
    value_col='tasks',
    bin_color='green', 
    space_labels_on=True,
    animation_idx=None,
    edges_on=False,
    nodes_on=False,
    edge_labels_on=False, 
    node_labels_on=False,
    node_size=30, 
    scale=8,
    x_scale=None,
    y_scale=None,
    use_edge_weights=False,
    display_result=True,
    save_image_path="",
):
    G = parse_domain_input(domain_id_or_graph)

    animated = animation_idx is not None
    fig, ax = format_plot(x_scale, y_scale, scale)
        
    (node_pos, 
    node_colors, 
    node_sizes, 
    weights, ) = get_plot_data(
        G,
        use_edge_weights, 
        node_size, 
        edges_on,
        nodes_on
    )

    tasks, total_tasks = parse_pick_data(reslot_algo_output, location_col, post_loc_col, value_col)
    dc_dict, domain, most_popular = build_dc_dict(G, tasks) #alphas)
    ax = draw_rectangles(dc_dict, ax, domain, most_popular, bin_color, space_labels_on)

    nx.draw(
        G, 
        ax=ax,
        with_labels=node_labels_on, 
        node_size=node_sizes, 
        pos=node_pos,
        edge_color=EDGE_COLOR,
        node_color=node_colors, 
        width=weights
    )

    if edge_labels_on:
        nx.draw_networkx_edge_labels(G, node_pos, ax=ax)
            
    max_percent = most_popular/total_tasks*100
    add_legend(bin_color, max_percent)
    
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
