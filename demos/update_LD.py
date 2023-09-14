from opt.io.local import get_domain_graph, update_warehouse_graph
from opt.graph import add_shortest_paths, add_distances



# Load existing graph
LD = get_domain_graph("LD")

# Add mocked-up 'delivery zone'
DZ_NAME = 'Delivery zone'
LD.bins[DZ_NAME] = {'pos': [49.1566650390625, -13.102465629577637, 1.4], 'closest_waypoint': 1280, 'distance': 0.0}

# Add missing (important) edges
LD.add_edge(1038, 1277, weight=3.267545019288563)
LD.add_edge(1282, 1283, weight=0.9252704671223881)
LD.add_edge(1284, 1285, weight=2.528270467122397)
# Update shortst paths and distances
add_shortest_paths(LD)
add_distances(LD)

update_warehouse_graph("LD", LD)
