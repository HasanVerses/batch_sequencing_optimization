import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import networkx as nx
import pickle


def json_to_graph(json_file):
    if json_file.get("format") == "node_link":
        nx_data = json_file.get("nx_data")
        if nx_data:
            G = nx.node_link_graph(nx_data)
        else:
            print("The 'nx_data' key is missing or empty in the JSON file.")
    else:
        print("The JSON format is not 'node_link'.")

    return G


def add_shortest_paths(graph):
    """Add shortest paths between all points (computed using Dijkstra's algorithm) as a property to the input graph."""
    graph.shortest_paths = {k: v for (k, v) in list(nx.all_pairs_dijkstra_path(graph))}


def add_distances(graph):
    """Add shortest distances (computed using Dijkstra's algorithm) as a property to the input graph."""
    graph.distances = {k: v for (k, v) in list(nx.all_pairs_dijkstra_path_length(graph))}

def add_bins(graph, json_file):
    """Add bins as a property to the input graph."""
    bin_dict = {}
    for bin, attributes in json_file['bins'].items():
        bin_dict[bin] = attributes
    graph.bins = bin_dict


def create_graph_from_json(json_path):
    print("Loading and updating graph...")
    with open(json_path, 'r') as file:
        # Parse the JSON data
        graph_json = json.load(file)

    graph = json_to_graph(graph_json)

    add_bins(graph, graph_json)

    # Manually add 'delivery zone' at designated location
    DZ_NAME = 'Delivery zone'
    graph.bins[DZ_NAME] = {'pos': [49.1566650390625, -13.102465629577637, 1.4], 'closest_waypoint': 1280, 'distance': 0.0}

    # Manually add missing edges near delivery zone
    graph.add_edge(1038, 1277, weight=3.267545019288563)
    graph.add_edge(1282, 1283, weight=0.9252704671223881)
    graph.add_edge(1284, 1285, weight=2.528270467122397)
    add_shortest_paths(graph)
    add_distances(graph)

    return graph


