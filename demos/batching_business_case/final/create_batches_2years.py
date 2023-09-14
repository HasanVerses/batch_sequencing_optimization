import pandas as pd
import datetime
from tqdm import tqdm

from opt.analysis import batch_assignments, convert_cols_to_datetime, filter_nri_pick_data
from opt.io.local import get_domain_graph



### Create batches with fixed target batch sizes 5-40 ###

print("Loading LD graph...")
LD = get_domain_graph("LD")
DZ_waypoint = LD.bins['Delivery zone']['closest_waypoint']
dz_bin = 'Delivery zone'

print("Loading data...")
# FULL_DATA_PATH = 'data/LD-DATA-05312022.csv'
DATA_PATH = 'data/to_sequence_all_ecomm.csv'
# data = filter_nri_pick_data(pd.read_csv(FULL_DATA_PATH), min_assignment_length=None, graph_to_filter_by=LD)
# data.to_csv('to_sequence_all_ecomm.csv')
data = pd.read_csv(DATA_PATH)

data = convert_cols_to_datetime(data)

by_date = data.groupby([data['AssignmentEndTime'].dt.date])

min_size = 5
max_size = 40
step_size = 5
SIZE_RANGE = range(min_size, max_size+step_size, step_size)

OUTPUT_FOLDER = 'data/batches_of_size_2years'

for size in SIZE_RANGE:
    print(f"Creating batches with target size {size}...")
    for date, group in tqdm(by_date):
        print("Date: ", date)
        batch_assignments(
            group, 
            date, 
            LD, 
            constraint_param=size,
            min_size=None,
            delivery_zone=dz_bin,
            output_name=f'size{size}',
            output_folder=OUTPUT_FOLDER + f'/size{size}',
            plot_energies=True,
            plot_graph=False
        )

