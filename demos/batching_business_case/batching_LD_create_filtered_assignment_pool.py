import pandas as pd
from opt.io.local import get_domain_graph



MAIN_DATA_PATH = 'data/LD-DATA-05312022.csv'
SUBSET_DATA_PATH = 'data/snaking_assignment_subset.csv'

print("Loading data...")
data = pd.read_csv(MAIN_DATA_PATH)
selected_subset_ids = pd.read_csv(SUBSET_DATA_PATH)

LD = get_domain_graph("LD")
assert 'Delivery zone' in LD.bins.keys(), "Delivery zone missing!" # Test for updated LD graph

# Filter data by assignment length and 'Ecom' status
filtered = data[data['AssignmentNumber'].isin(selected_subset_ids['assignmentnumber'])].copy()
print(f"Filtered data contains {len(filtered)} / {len(data)} tasks")

## Filter by presence in LD ##
print("Filtering by presence in LD...")
# Group by assignment
by_assignment_nos = filtered.groupby(filtered['AssignmentNumber']).groups
on_LD_ids = [k for k in by_assignment_nos if 
    all(filtered.loc[by_assignment_nos[k]]['FromLocation'].isin(list(LD.bins.keys())))
]
to_sequence = filtered[filtered['AssignmentNumber'].isin(on_LD_ids)].copy()
print(f"Filtered-by-LD data contains {len(to_sequence)} / {len(filtered)} tasks")

# Group by date and assignment no.
to_sequence.loc[:,('AssignmentStartTime')] = pd.to_datetime(to_sequence.loc[:,('AssignmentStartTime')]).copy()
to_sequence.loc[:,('AssignmentEndTime')] = pd.to_datetime(to_sequence.loc[:,('AssignmentEndTime')]).copy()
to_sequence.to_csv("data/LD_to_sequence.csv")
