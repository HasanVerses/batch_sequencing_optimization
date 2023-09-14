import pandas as pd
import time
import datetime
from tqdm import tqdm

from opt.io.local import get_domain_graph
from opt.model.optimizer import Optimizer
from opt.model.constraint import ConstraintFn
from opt.model.modifier import ModifierFn
from opt.defaults.cost import GraphTaskClusterCost, AssignmentMSECost, numClusterCost
from opt.defaults.modifier import ClusterSwapSmartTaskStochastic, ClusterSwapSmartTask
from opt.defaults.constraint import min_cluster_size, max_k_between, max_k_equals
from opt.algorithms.kmedoids import initialize_task_assignments
from opt.visual.visual import plot_graph_with_cluster_ids, plot_overlaid_assignment_vectors



LD = get_domain_graph("LD")

DATA_PATH = 'data/LD_to_sequence.csv'
data = pd.read_csv(DATA_PATH)
data.loc[:,('AssignmentStartTime')] = pd.to_datetime(data.loc[:,('AssignmentStartTime')]).copy()
data.loc[:,('AssignmentEndTime')] = pd.to_datetime(data.loc[:,('AssignmentEndTime')]).copy()
data.loc[:,('TaskStartDate')] = pd.to_datetime(data.loc[:,('AssignmentStartTime')]).copy()
data.loc[:,('TaskEndDate')] = pd.to_datetime(data.loc[:,('AssignmentEndTime')]).copy()

by_date = data.groupby([data['AssignmentEndTime'].dt.date])

for date, group in tqdm(by_date):
    print(f"Processing {date}...")
    print(date, group['FromLocation'].value_counts())
    task_locations = group[['FromLocation', 'TaskEndDate']].copy()
    task_locations.to_csv(f"data/batching_test_2/LD_tasks_{date}.csv")
    tasks = list(task_locations['FromLocation'])

    ref_vec = list(group.groupby("AssignmentNumber").size())
    num_assignments = len(ref_vec)
    print("Vector of assignment lengths: ", ref_vec)

    min_size= 3
    lower_upper = [num_assignments - 2, num_assignments + 2]

    print("Initializing cluster assignments...")
    cluster_assigns, task_distance_matrix, from_dist_map, inverse_map, to_idx_map = \
        initialize_task_assignments(LD, group, max_k = num_assignments, from_df=True)

    # cluster_assigns, task_distance_matrix, from_dist_map, inverse_map, to_idx_map = \
    #     initialize_task_assignments(LD, tasks, max_k = num_assignments, use_bins=True)

    print("Initial clusters", [len(x) for x in cluster_assigns])
    print(f"k = {len(cluster_assigns)}")
    distanceCost = GraphTaskClusterCost(task_distance_matrix)

    mseCost = AssignmentMSECost(assignment_size_vec = ref_vec, scale_fn=lambda x: x*1000)
    clusterCost = numClusterCost(num_assignments)


    minClustSize = ConstraintFn(min_cluster_size, [min_size], hard = True)
    #nClustConstraint = ConstraintFn(max_k_between, [lower_upper], hard = True)
    nClustConstraint = ConstraintFn(max_k_equals, [num_assignments], hard = True)

    opt = Optimizer(costs = [distanceCost, clusterCost, mseCost],
                    cost_weights = [1.0, 0.2, 5.0],  
                    constraints = [minClustSize, nClustConstraint],
                    modifier = ClusterSwapSmartTask(task_distance_matrix),
                    energy_plot_path=f'data/batching_test_2/energies_{date}.png'
                    )
                    
    start = time.time()
    result = opt.optimize(cluster_assigns)
    duration = time.time() - start
    result_mapped = [[to_idx_map[from_dist_map[n]] for n in r] for r in result]

    inverse_map = {v: k for k, v in to_idx_map.items()}
    df_indices = [[inverse_map[x] for x in r] for r in result_mapped]
    verses_batches = []
    for idx, r in enumerate(result_mapped):
        df_indices = [inverse_map[x] for x in r]
        verses_batch = group.loc[df_indices].copy()
        verses_batch['BatchingAssignmentNumber'] = idx
        verses_batch['ClusteringTime'] = duration
        verses_batches.append(verses_batch)

    # Save created assignments (batches)
    verses_batches = pd.concat(verses_batches)
    verses_batches.to_csv(f"data/batching_test_2/LD_verses_batches_{date}.csv")

    print("result:", result_mapped)
    print("optimized distance: ", distanceCost.eval(result))
    print("optimized number of clusters: ", len(result))

    assign_vec = [len(x) for x in result]
    print("number of assignments in result", sum(ref_vec))

    # Plot and save results
    plot_graph_with_cluster_ids(LD, result_mapped, vary_sizes=True, color_skew_factor=20, save_path=f'data/batching_test_2/graph_{date}.png', display=False)
    plot_overlaid_assignment_vectors(ref_vec, assign_vec, save_path=f'data/batching_test_2/distribution_{date}.png', sort=True, display=False)
