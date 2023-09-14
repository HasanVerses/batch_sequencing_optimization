import os

from opt.io.local import load_graph, store_graph, normalize_folder, normalize_fn
from opt.graph import random_graph


def test_store_graph():
    graph_data = store_graph(G, filename, folder)

    assert "nx_data" in graph_data
    assert "shortest_paths" in graph_data
    assert "distances" in graph_data
    assert "dimensions" in graph_data
    assert os.path.exists(path)

def test_load_graph():
    G_loaded = load_graph(filename, folder)
    assert list(G_loaded.nodes()) == list(G.nodes())
    assert list(G_loaded.edges()) == list(G_loaded.edges())
    assert G_loaded.shortest_paths == G.shortest_paths
    assert G_loaded.distances == G.distances
    assert G_loaded.dimensions == G.dimensions

    # Clean up test files
    try:
        os.remove(path)
        os.rmdir(folder)
    except:
        pass


G = random_graph()

filename = "random_graph.json"
folder = "test_storage"
path = f"{normalize_folder(folder)}{normalize_fn(filename, '.json')}"
