from opt.algorithms.common import convert_duplicate_node_multi
from opt.algorithms.reslotting import source_of, decode_destination_nodes, corresponding_node, is_destination
from opt.graph import get_closest_waypoint_multi, get_shortest_path_multi, get_distance_multi



class Sequence():
    def __init__(self, raw_data, graph, use_bins=True, cart_capacity=None, end=None):
        self.raw = raw_data
        self.graph = graph
        self.use_bins = use_bins

        self.decoded = convert_duplicate_node_multi(decode_destination_nodes(raw_data))
        self.sequence, self.path, self.indices = sequence_dict_list(graph, raw_data, use_bins)
        self.english = [in_english(location) for location in raw_data]
        self.waypoints = self._get_waypoints(graph) if use_bins else self.decoded
        self.distance = get_distance_multi(graph, self.decoded, use_bins)
        self.end = end or self.waypoints[-1]
        
        self.sources, self.destinations = [], []
        for n in self.sequence:
            if n['type'] == 'source':
                self.sources.append(n)
            elif n['type'] == 'destination':
                self.destinations.append(n)
        
        self.for_reslotting = len(self.destinations) > 0
        assert (cart_capacity is not None) if self.for_reslotting else True, "Must supply `cart_capacity` for reslotting sequence!"
        if self.for_reslotting:
            self.swap_dict = {s['location']: d['location'] for s, d in zip(self.sources, self.destinations)}
        self.cart_capacity = cart_capacity
    
    def path_slice(self, path_idx):
        if path_idx is None:
            return {
                'sources': [x.get('location') for x in self.sources], 
                'destinations': [x.get('location') for x in self.destinations], 
                'remaining_sources': [], 
                'remaining_destinations': []
            }
        return {
            'sources': [x.get('location') for x in self.sources if x['path_idx'] <= path_idx],
            'destinations': [x.get('location') for x in self.destinations if x['path_idx'] <= path_idx],
            'remaining_sources': [x.get('location') for x in self.sources if x['path_idx'] > path_idx],
            'remaining_destinations': [x.get('location') for x in self.destinations if x['path_idx'] > path_idx]
        }

    def waypoints_slice(self, path_idx):
        if path_idx is None:
            return self.waypoints, None
        else:
            before, after = [], []
            for idx, x in enumerate(self.waypoints):
                if self.indices[idx] <= path_idx:
                    before.append(x)
                elif self.indices[idx] > path_idx:
                    after.append(x)
            
            return before, after

    def get_locations(self):
        return self.sources, self.destinations
    
    def add_baseline(self, distance):
        self.baseline = distance
        
    def _get_waypoints(self, graph):
        return get_closest_waypoint_multi(graph, self.decoded)

    def __list__(self):
        return self.sequence
    
    def __len__(self):
        return len(self.sequence)
    
    def __eq__(self, other):
        return (self.sequence == other.sequence) and (self.location_type == other.location_type)
    
    def __bool__(self):
        return len(self.sequence) > 0
    
    def __str__(self):
        return str(self.raw)
    
    def __getitem__(self, key):
        return self.sequence[key]
    
    def __iter__(self):
        return (x for x in self.sequence)


def in_english(node_id, suffix='_d.'):
    if is_destination(node_id):
        return f"Place item from {source_of(node_id, suffix)} in {corresponding_node(node_id, suffix)}"
    else:
        return f"Pick item from {node_id}"


def sequence_dict_list(graph, sequence, use_bins, suffix='_d.', prefix='_o.'):
    decoded_sequence = convert_duplicate_node_multi(
        decode_destination_nodes(sequence, suffix, prefix)
    )
    sequence_waypoints = get_closest_waypoint_multi(graph, decoded_sequence) if use_bins else decoded_sequence
    path, sequence_indices = get_shortest_path_multi(graph, sequence_waypoints, return_destination_indices=True)

    seq_dict_list = [{'location': corresponding_node(n, suffix, prefix) if is_destination(n, suffix) else n, 
                'type': 'destination' if is_destination(n, suffix) else 'source',
                'path_idx': sequence_indices[idx]
            } for idx, n in enumerate(sequence)]
    
    if use_bins:
        for idx in range(len(sequence)):
            seq_dict_list[idx]['bin'] = sequence[idx]
    
    return seq_dict_list, path, sequence_indices
