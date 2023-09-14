import pandas as pd
from tqdm import tqdm

from opt.analysis import batch_assignments, convert_cols_to_datetime
from opt.io.local import get_domain_graph



### Create batches with fixed target batch sizes 5-40 ###

DATA_PATH = 'data/LD_to_sequence.csv'
data = convert_cols_to_datetime(pd.read_csv(DATA_PATH))

LD = get_domain_graph("LD")
DZ_waypoint = LD.bins['Delivery zone']['closest_waypoint']
dz_bin = 'Delivery zone'

by_date = data.groupby([data['AssignmentEndTime'].dt.date])

min_size = 5
max_size = 40
step_size = 5
SIZE_RANGE = range(min_size, max_size+step_size, step_size)

OUTPUT_FOLDER = 'data/batches_of_size'


for size in SIZE_RANGE:
    print(f"Creating batches with target size {size}...")
    for date, group in tqdm(by_date):
        print("Date: ", date)
        batch_assignments(
            group, 
            date, 
            LD, 
            constraint_param=size,
            delivery_zone=dz_bin,
            output_name=f'size{size}',
            output_folder=OUTPUT_FOLDER + f'/size{size}',
            plot_energies=True,
            plot_graph=False
        )

