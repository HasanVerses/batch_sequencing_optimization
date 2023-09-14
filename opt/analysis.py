import time
import glob
import os
import datetime
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Union

from opt.io.local import get_domain_graph, normalize_folder
from opt.api.snaking import compare_to_optimized, optimize_picks
from opt.algorithms.kmedoids import initialize_task_assignments
from opt.model.optimizer import Optimizer
from opt.model.constraint import ConstraintFn
from opt.defaults.modifier import ClusterSwapSmartTask, SwappingMergingMix
from opt.defaults.constraint import min_cluster_size, max_k_equals
from opt.defaults.cost import GraphTaskClusterCost, AssignmentMSECost, numClusterCost, GraphTaskClusterCost_ExtraNode_v2, ClusterSizeCost
from opt.visual.visual import plot_graph_with_cluster_ids, plot_overlaid_assignment_vectors



def convert_cols_to_datetime(
    data, 
    cols=['AssignmentStartTime', 'AssignmentEndTime', 'TaskStartDate', 'TaskEndDate']
):
    for col in cols:
        data.loc[:,(col)] = pd.to_datetime(data.loc[:,(col)]).copy()

    return data

def add_assignment_lengths(df, groupby_col="AssignmentNumber", index="Unnamed: 0", col_name="AssignmentLength"):
    """ 
    Adds another column to an existing dataframe `df` that simply lists the size of the batch or assignment (the number
    of tasks in the assignment, aka "AssignmentLength")
    """

    df[col_name] = df.groupby([groupby_col])[index].transform(len).copy()
    return df

def filter_assignments_by_value(df, target_col, target_val, groupby_col="AssignmentNumber", verbose=True):
    if verbose:
        print(f"Filtering by `{target_val}`")
        print(f"Pre-filtered length: {len(df)}")
    return df.groupby(groupby_col).filter(lambda x: (x[target_col] == target_val).all())

def filter_by_ecomm(df, **kwargs):
    return filter_assignments_by_value(df, target_col="SubDocumentType", target_val="Ecom", **kwargs)

def filter_by_completed(df, **kwargs):
    return filter_assignments_by_value(df, target_col="TaskStatusDescription", target_val="Completed", **kwargs)

def filter_by_min_assignment_length(df, min_assignment_length, length_col="AssignmentLength", verbose=True):
    if verbose:
        print(f"Filtering by minimum assignment length {min_assignment_length}...")
        print(f"Pre-filtered length: {len(df)}")
    if length_col not in df.columns:
        df = add_assignment_lengths(df)
    return df[df["AssignmentLength"] >= min_assignment_length].copy()

def filter_by_max_assignment_length(df, max_assignment_length, length_col="AssignmentLength", verbose=True):
    if verbose:
        print(f"Filtering by minimum assignment length {max_assignment_length}...")
        print(f"Pre-filtered length: {len(df)}")
    if length_col not in df.columns:
        df = add_assignment_lengths(df)
    return df[df["AssignmentLength"] <= max_assignment_length].copy()

    
def filter_nri_pick_data(
    data: pd.DataFrame, 
    by_ecomm: bool=True,
    by_completed: bool=True, 
    min_assignment_length: Union[int, None]=3, 
    max_assignment_length: Union[int, None]=None, 
    graph_to_filter_by: Union[nx.Graph, str]=None,
    include_assignment_lengths=True,
    csv_output_path: Union[str, None]=None,
    verbose: bool=True
):
    """
    Apply filters to select a subset of data to analyze from a pool of NRI pick assignments.

    Arguments:
    ==========
    `data`: DataFrame containing the pick data (presumably raw pick data from NRI)
    `by_ecomm`: Toggles whether to filter by `SubDocumentType` == `Ecom` (ecommerce picks only)
    `by_completed`: Toggles whether to filter by `TaskStatusDescription` == `Complete`
    `min_assignment_length`: Minimum length (in terms of # of picks) of assignments to include in the analysis
    `max_assignment_length`: Max ""  ""  ""
    `graph_to_filter_by`: Graph (or domain ID) used to filter data: only assignments all of whose picks are
        from locations stored in the graph data will be included
    `add_assignment_lengths`: Toggles whether to add an `AssignmentLength` column to the input DataFrame that
        records the length of the assignment in the `AssignmentNumber` column for each row. This is added anyway
        if min/max assignment length constraints are used.
    `csv_output_path`: Full path to output .csv file
    `verbose`: Toggles printing status to console
    Returns:
    ==========
    `data`: The filtered version of the original input DataFrame (filtered by a mixture of criteria, determined by the values of the other input arguments).
    """
    if 'AssignmentLength' not in data.columns and include_assignment_lengths:
        data = add_assignment_lengths(data)
    
    if by_ecomm:
        data = filter_by_ecomm(data, verbose=verbose)

    if by_completed:
        data = filter_by_completed(data, verbose=verbose)

    if min_assignment_length is not None:
        data = filter_by_min_assignment_length(data, min_assignment_length)

    if max_assignment_length is not None:
        data = filter_by_max_assignment_length(data, max_assignment_length)

    if graph_to_filter_by is not None:
        print("Filtering by presence in graph...")
        if type(graph_to_filter_by) == str: # In case a domain ID is passed in instead
            graph_to_filter_by = get_domain_graph(graph_to_filter_by)

        print(f"Pre-filtered length: {len(data)}")        
        data = data.groupby(["AssignmentNumber"]).filter(
            lambda x: (x['FromLocation'].isin(list(graph_to_filter_by.bins.keys())).all())
        )    
    
    print("Filtered data length: ", len(data))    
    if csv_output_path is not None:
        data.to_csv(csv_output_path)

    return data


def run_sequencing_test(
    assignment_data: pd.DataFrame, 
    graph: nx.Graph, 
    n: int=1,
    num_steps: int=40,
    algorithm: str='annealing', 
    save_results: bool=False,
    dz_name='Delivery zone', # Temporary delivery zone representation in LD graph
    batch_field = 'AssignmentNumber', # NRI assignment data default
    location_field = 'FromLocation' # NRI assignment data default
):
    """
    Runs a test comparing optimized to orgiinal sequence length for a given assignment, given NRI-style pick data.

    Arguments:
    ==========
    `assignment_data`: Pandas dataframe, presumably in the format NRI uses to output pick data
    `graph`: Graph on which to run the tests
    `n`: Number of parallel runs of the optimization algorithm
    `num_steps`: Number of outer loop iterations for the optimization algorithm
    `algorithm`: type of optimization algorithm to run
    `save_results`: Toggles whether to save these individual assignment results as a .csv or simply return them
    `dz_name`: Name of placeholder 'delivery zone' location in the warehouse graph data
    `batch_filed`: DataFrame column containing assignmnet numbers
    `location_field`: DataFrame column containing pick locations

    Returns:
    ==========
    `results`: A dictionary that can be used to update a dataframe containing aggregate analyses of pick data, 
    including execution time for the optimization
    `sequence`: the original (unoptimized) sequence
    `opt_sequence`: the new (optimized) sequence
    """
    assignment_data.reset_index(drop=True)

    assignment_no = next(iter(assignment_data[batch_field]))
    assignment = assignment_data[location_field]

    print(f"Processing assignment {assignment_no}")
    pick_locations = list(assignment)
    assignment_length = len(pick_locations)
    sequence = [dz_name] + pick_locations + [dz_name]

    start = time.time()
    r = compare_to_optimized(
        graph,
        sequence,
        n=n,
        algorithm=algorithm,
        num_steps=num_steps,
        fixed_start=True,
        fixed_end=True,
        cumulative_distances=False,
        output_df=False
    )
    duration = time.time() - start

    if not r:
        return None, None, None

    unopt = r['original']['distance']
    opt = r['optimized']['distance']
    diff = unopt - opt

    opt_sequence = r['optimized']['sequence']

    results = {
        'AssignmentNumber': assignment_no,
        'AssignmentLength': assignment_length,
        'UnoptimizedSequence': '|'.join(sequence),
        'OptimizedSequence': '|'.join(opt_sequence),
        'UnoptimizedDistance': unopt,
        'OptimizedDistance': opt,
        'DistanceSavings': diff,
        'SavingsPercent': diff/unopt,
        'ExecutionTime': duration,
        'NumParallelRuns': n,
        'Algorithm': algorithm 
    }
    if save_results:
        results = pd.DataFrame(results, index=[0])
        results.to_csv(f'LD_snaking_{assignment_no}.csv')

    return results, sequence, opt_sequence


def sequence_batched_tasks(assignment_data, graph, n=1, algorithm='annealing', save_results=False, date=None, delivery_zone_name='Delivery zone'):
    """
    This function a previously batched set of tasks (e.g. a DF) and sequences it.

    Arguments:
    ==========
    `assignment_data`: Pandas dataframe representing a single assignment, with a new `BatchingAssignmentNumber` 
    column.
    `graph`: Graph on which to run the tests
    `n`: Number of parallel runs of the optimization algorithm
    `algorithm`: type of optimization algorithm to run (e.g. 'annealing')
    `save_results`: Toggles whether to save these individual assignment results as a .csv or simply return them
    `date`: date on which the assignment of tasks was issued.
    `delivery_zone_name`: Name of placeholder 'delivery zone' location in the warehouse graph data

    Returns:
    ==========
    `results`: A dictionary that can be used to update a dataframe containing aggregate analyses of pick data, 
    including execution time for the optimization
    `seq`: the new (optimized) sequence
    """
    assignment_no = next(iter(assignment_data['BatchingAssignmentNumber']))
    assignment = assignment_data['FromLocation']

    print(f"Processing assignment {assignment_no}")
    pick_locations = list(assignment)
    print("Locations: ", pick_locations)
    assignment_length = len(pick_locations)
    sequence = [delivery_zone_name] + pick_locations + [delivery_zone_name]

    start = time.time()
    distance, seq, path = optimize_picks(
        graph,
        sequence,
        n=n,
        algorithm=algorithm,
        num_steps=40,
        fixed_start=True,
        fixed_end=True,
        cumulative_distances=False,
        output_df=False
    )
    duration = time.time() - start

    if not distance:
        return None, None, None

    results = {
        'Date': date,
        'BatchAssignmentNumber': assignment_no,
        'BatchAssignmentLength': assignment_length,
        'BatchOptimizedSequence': '|'.join(seq),
        'BatchOptimizedDistance': distance,
        'BatchExecutionTime': duration,
        'BatchNumParallelRuns': n,
        'BatchAlgorithm': algorithm 
    }
    if save_results:
        results = pd.DataFrame(results, index=[0])
        results.to_csv(f'LD_snaking_batching_{assignment_no}.csv')

    return results, seq

def distribution_constrained_optimizer(
    ref_vec,
    distance_cost, 
    min_size_constraint, 
    modifier, 
    energy_plot_path
):
    """
    Function that instantiates an `Optimizer` that uses a provided `ref_vec` of assignment sizes, a `distance_cost` and other constraints/inputs to generate
    an Optimizer with the desired joint constraints/costs. This is a particular variant that uses a mean-squared error (or L2) cost to match the distribution of
    optimized batch sizes, to a set of sizes (or assignment lengths) given by the entries of `ref_vec`.

    Arguments:
    ==========
    `ref_vec`: list of integers containing the sizes of a bunch of assignments from a particular time window (usually within a single day of NRI pick data).
    `distance_cost`: an instance of `CostFn`. Usually, we assume this is a cost function (e.g. one of those in `app.defaults.cost`) that measures the 
                    quality of a clustering solution using a distance metric computed across the clusters/batches of the solution (e.g. the average shortest path length within nodes of a cluster)
    `min_size_constraint`: an instance of `ConstraintFn` that implements a hard constraint on the minimum size of a cluster / assignment.
    `modifier`: an instance of `ModifierFn` that mutates or modifies the current configuration of cluster assignments to propose a new or "modified" one. 
    `energy_plot_path`: path for where to write a .png of the trajectory of the energy over the course of optimization. 
   
    Returns:
    ==========
    An instance of the `Optimizer` class with a customized cost that combines the distance cost, a cost on the total number of clusters, and 
    the `AssignmentMSECost` that encourages the distribution of cluster sizes in the solution to match the list of assignment lengths given by `ref_vec`.
    """
    print("Vector of assignment lengths: ", ref_vec)
    num_assignments = len(ref_vec)
    clusterCost = numClusterCost(num_assignments)
    mseCost = AssignmentMSECost(assignment_size_vec = ref_vec, scale_fn=lambda x: x*1000)    
    nClustConstraint = ConstraintFn(max_k_equals, [num_assignments], hard = True)

    constraints = [min_size_constraint, nClustConstraint] if min_size_constraint else [nClustConstraint]

    return Optimizer(costs = [distance_cost, clusterCost, mseCost],
            cost_weights = [1.0, 0.2, 5.0],  
            constraints = constraints,
            modifier = modifier,
            energy_plot_path=energy_plot_path
        )

def size_constrained_optimizer(
    target_size,
    distance_cost, 
    min_size_constraint, 
    modifier, 
    energy_plot_path
):
    """
    Function that instantiates an `Optimizer` that uses a provided `target_size`, a `distance_cost` and other constraints/inputs to generate
    an Optimizer with the desired joint constraints/costs. This is a particular variant that uses a mean-squared error (or L2) cost to penalize
    the size of each assignment/cluster, in proportion to how far it deviates from `target_size`.

    Arguments:
    ==========
    `target_size`: integer encoding the "target" size of each assignment / batch that the size of each cluster in the optimmized solution will be encouraged to move towards.
    `distance_cost`: an instance of `CostFn`. Usually, we assume this is a cost function (e.g. one of those in `app.defaults.cost`) that measures the 
                    quality of a clustering solution using a distance metric computed across the clusters/batches of the solution (e.g. the average shortest path length within nodes of a cluster)
    `min_size_constraint`: an instance of `ConstraintFn` that implements a hard constraint on the minimum size of a cluster / assignment.
    `modifier`: an instance of `ModifierFn` that mutates or modifies the current configuration of cluster assignments to propose a new or "modified" one. 
    `energy_plot_path`: path for where to write a .png of the trajectory of the energy over the course of optimization. 
    Returns:
    ==========
    An instance of the `Optimizer` class with a customized cost that combines the distance cost and a `ClusterSizeCost` 
    that encourages the average cluster size in the solution to match the number given by `target_size`.
    """
    size_cost = ClusterSizeCost(target_size)#, scale_fn=lambda x: x*1000)
    constraints = [min_size_constraint] if min_size_constraint else []
    return Optimizer(
            costs = [distance_cost, size_cost],
            cost_weights=[1., 10.],
            constraints = constraints,
            modifier=modifier,
            energy_plot_path=energy_plot_path
        ), size_cost


def assign_num_constrained_optimizer(num_assignments, distance_cost, modifier, energy_plot_path):
    nClusterCost = numClusterCost(num_assignments)
    nClustConstraint = ConstraintFn(max_k_equals, [num_assignments], hard = True)

    return Optimizer(costs = [distance_cost, nClusterCost],
            cost_weights = [1.0, 10.0],  
            constraints = [nClustConstraint],
            modifier = modifier,
            energy_plot_path=energy_plot_path
        ), nClusterCost

def batch_assignments(
    assignment_group: pd.DataFrame, 
    time_window: datetime.date,
    graph: nx.Graph,
    constraint_param='distribution',
    delivery_zone=None,
    output_name: Union[str, None]=None,
    output_folder='batching_test',
    location_col='FromLocation',
    min_size=3,
    plot_energies=False,
    plot_graph=True,
    plot_distribution=True,
    time_execution=True
):
    """
    Function that batches NRI assignments from a given time according to one of our (Verses') in-house batching algorithms.

    Arguments:
    ==========
    `assignment_group`: a pandas DataFrame containing the picks corresponding to a set of assignments from a given time (e.g. a day-long window of assignments).
    `time_window`: timw window from which the NRI assignments were pulled
    `graph`: a networkx Graph object representation of the warehouse in question
    `constraint_param`: either a `str` (`distribution`) or an `int` determining the type of cosntraint used to optimize the assignments. If `distribution`, then the vector 
                        of assignment lengths is used to generate a "reference vector" (`ref_vec`) that is used to pull the optimized cluster solution closer to the "observed" (or NRI-determined)
                        distribution of assignment sizes. If an `int, then this encodes a desired assignment size or batch size, which is used to enforce an MSE (least-squared error) cost
                        on the average size of a batch / assignment in the cluster solution, with no explicit "distribution-shaping" constraint.
    `delivery_zone`: the ID (in terms of NRI locations) of an optional delivery zone node, that will be used to augment the cost for each cluster by adding an additional "distance to delivery zone" to optimize the assignments.
    `output_name` : optional string that will be used to identify the analysis run in terms of both the .csv of outputs as well as the output energy plot
    `output_folder`:optional output folder name in which the results will be stored (default: 'batching_test')
    `location_col`: optional string (default: `"FromLocation"`) that denotes the column of the `assignment_group` dataframe, which is used to do ... @NOTE: @exilefaker can you hop in an finish this docstring, I don't understand function of this column.
    `plot_energies`: optional flag (default `False`) for whether to plot the energy over the course of optimization
    `plot_graph`: optional flag (default: `True`) for whether to plot the spatial graph of the warehouse with relative locations / assignments overlaid at the end of the optimization.
    `plot_distribution`: optional flag (default: `True`) for whether to plot the distribution of assignment sizes found by the cluster optimization, on top of the assignment size distribution given in the NRI-provided assignments AKA `assignment_group`.
                        @NOTE: This distributional overlay is only plotted if `constraint_param == "distribution"`, i.e. the actual distribution of assignment sizes is also used to optimize the clustering solution.
    `time_execution`: optional flag (default: `True`) for whether to time the optimization of the clustering solution and save it.
    Returns:
    ==========
    `verses_batches`: a pandas DataFrame containing information about the found batching optimization solution, such as an ordinal index assigned to each optimized cluster in the warehouse graph
    (stored in the column "BatchingAssignmentNumber") as well as the node ids of the locations in each assignment (stored in the column `FromLocation`). These results are also
    converted and written to disk in the form of a .csv with the naming convention "{`output_folder`}{`output_name`}_batches_{`time_window`}.csv"
    """
    output_folder = normalize_folder(output_folder)
    if output_name is None:
        output_name = output_folder
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Processing {time_window}...")
    print(time_window, assignment_group[location_col].value_counts()) # what's going on with this line?

    minClustSize = None if min_size is None else ConstraintFn(min_cluster_size, [min_size], hard = True)
    #minClustCost = CostFn(lambda x: x )
    
    ref_vec = list(assignment_group.groupby("AssignmentNumber").size())
    num_assignments = len(ref_vec)

    print("Initializing cluster assignments...")

    cluster_assigns, task_distance_matrix, linear_idx_reduced_to_bin, bin_to_linear_idx_reduced, _, _ = \
        initialize_task_assignments(
            graph, 
            list(assignment_group['FromLocation']), 
            max_k=num_assignments, 
            use_bins=True, 
            extra_node=delivery_zone
        )
    print("Initial clusters", [len(x) for x in cluster_assigns])
    print(f"k = {len(cluster_assigns)}")

    distanceCost = GraphTaskClusterCost_ExtraNode_v2(task_distance_matrix, extra_node_id=bin_to_linear_idx_reduced[delivery_zone])
    modifier = SwappingMergingMix(task_distance_matrix)

    energy_plot_path = f'{output_folder}{output_name}_{time_window}_energy.png' if plot_energies else None

    if constraint_param == 'distribution':
        opt = distribution_constrained_optimizer(ref_vec, distanceCost, minClustSize, modifier, energy_plot_path)

    elif constraint_param == 'number':
        opt, nClusterCost = assign_num_constrained_optimizer(num_assignments, distanceCost, modifier, energy_plot_path)

    elif type(constraint_param) == int:
        opt, sizeCost = size_constrained_optimizer(constraint_param, distanceCost, minClustSize, modifier, energy_plot_path)
                    
    if time_execution:
        start = time.time()
    result = opt.optimize(cluster_assigns)
    if time_execution:
        duration = time.time() - start
    result_mapped = [[linear_idx_reduced_to_bin[n] for n in r] for r in result]

    verses_batches = []
    for idx, r in enumerate(result_mapped):
        verses_batch = pd.DataFrame({'BatchingAssignmentNumber': idx, 'FromLocation': r, 'Date': time_window})
        if time_execution:
            verses_batch['ClusteringTime'] = duration
        verses_batches.append(verses_batch)

    # Save created assignments (batches)
    verses_batches = pd.concat(verses_batches)
    verses_batches.to_csv(f'{output_folder}{output_name}_batches_{time_window}.csv')
    
    print("optimized distance: ", distanceCost.eval(result))
    if type(constraint_param) == int: # Check for cluster size cost fn type
        print("optimized cluster size cost", sizeCost.eval(result))
    elif constraint_param=='number': # Check for assignment size constraint
        print("optimized assignment num cost", nClusterCost.eval(result))
    print("optimized number of clusters: ", len(result))

    assign_vec = [len(x) for x in result]
    print("number of assignments in result", sum(ref_vec))

    # Plot and save results
    if plot_graph:
        plot_graph_with_cluster_ids(
            graph, 
            result_mapped, 
            vary_sizes=True, 
            color_skew_factor=20, 
            save_path=f'{output_folder}{output_name}_graph_{time_window}.png', display=False
        )
    if plot_distribution and (constraint_param == 'distribution'):
        plot_overlaid_assignment_vectors(
            ref_vec, 
            assign_vec, 
            save_path=f'{output_folder}{output_name}_distribution_{time_window}.png', sort=True, display=False
        )

    return verses_batches


def analyze_sequenced_assignments(data_folder, output_name, output_folder='.', min_size=0, max_size=45, bin_size=5, plot_data=True):
    """
    Run analysis on a folder of .csv files containing data for potentially multiple days of optimized pick data.
    `data_folder`: Location of one or more .csv files to analyze in the aggregate
    `output_name`: Unique string used along with `output_folder` to create paths for output data
    `output_folder`: Folder in which to store output results
    `min_size`: Smallest assignment size to use when creating bins
    `max_size`: Largest assignment size to use when creating bins
    `bin_size`: Size of assignment bins to use in analysis
    `plot_data`: Toggles whether to save plots of aggregate results
    """
    
    days = [pd.read_csv(fn) for fn in glob.glob(f'{data_folder}/*.csv')]
    num_days = len(days)
    assert num_days > 0, "No data to analyze!"
    total = pd.concat(days, ignore_index=True)

    output_folder = normalize_folder(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        print(f"Warning: output path {output_folder} already exists.")
    output_name = output_name.split(".")[0]

    data = {
        'AssignmentLength': [],
        'TotalUnoptimizedDistance': [],
        'TotalOptimizedDistance': [],
        'MeanSavingsPercent': [],
        'DistanceSaved': [],
        'ExecutionTime': [],
        'NumberOfAssignments': []
    }

    size_range = range(min_size, max_size, bin_size)
    for size in size_range:
        assignments_of_len = total[total['AssignmentLength'].ge(size) & total['AssignmentLength'].le(size+bin_size)].copy()    

        data['TotalUnoptimizedDistance'].append(assignments_of_len['UnoptimizedDistance'].sum())
        data['TotalOptimizedDistance'].append(assignments_of_len['OptimizedDistance'].sum())
        data['MeanSavingsPercent'].append((assignments_of_len['DistanceSavings'].sum()/assignments_of_len['UnoptimizedDistance'].sum())*100)
        data['ExecutionTime'].append(assignments_of_len['ExecutionTime'].mean())
        data['AssignmentLength'].append(f"{size}-{size+bin_size}")
        data['NumberOfAssignments'].append(len(assignments_of_len))
        data['DistanceSaved'].append(assignments_of_len['DistanceSavings'].sum())

    longest_assignments = total[total['AssignmentLength'].ge(max_size)].copy() 

    data['TotalUnoptimizedDistance'].append(longest_assignments['UnoptimizedDistance'].sum())
    data['TotalOptimizedDistance'].append(longest_assignments['OptimizedDistance'].sum())
    data['MeanSavingsPercent'].append((longest_assignments['DistanceSavings'].sum()/longest_assignments['UnoptimizedDistance'].sum())*100)
    data['ExecutionTime'].append(longest_assignments['ExecutionTime'].mean())
    data['AssignmentLength'].append(f">{max_size}")
    data['NumberOfAssignments'].append(len(longest_assignments))
    data['DistanceSaved'].append(longest_assignments['DistanceSavings'].sum())

    results = pd.DataFrame(data)
    results.to_csv(f'{output_folder}{output_name}_aggregate.csv')

    if plot_data:
        # TODO: Perhaps move to app.visual

        plt.plot(data['AssignmentLength'], data['MeanSavingsPercent'])
        plt.title("VERSES sequencing: Mean distance savings by assignment length")
        plt.xlabel('Assignment length')
        plt.ylabel('Savings %')
        total_savings = (results['DistanceSaved'].sum()/results['TotalUnoptimizedDistance'].sum())*100
        plt.annotate(f"Total savings %: {total_savings:.2f}", (3,4))
        plt.annotate(f"Total distance saved: {results['DistanceSaved'].sum()/1000:.2f} km", (3,7))
        plt.annotate(f"Total number of assignments: {len(total)}", (3, 13))
        plt.annotate(f"Over {num_days} days", (3, 10))
        plt.savefig(f'{output_folder}{output_name}_aggregate_savings.png')
        plt.clf()

        plt.plot(data['AssignmentLength'], data['ExecutionTime'])
        plt.title("VERSES sequencing: Mean execution time by assignment length")
        plt.xlabel('Assignment length')
        plt.ylabel('Execution time (seconds)')
        plt.annotate(f"Avg. execution time: {results['ExecutionTime'].mean():.2f} seconds", (1,8))
        plt.annotate(f"Total number of assignments: {len(total)}", (1, 10))
        plt.savefig(f'{output_folder}{output_name}_aggregate_times.png')
        plt.clf()

        plt.bar(range(len(size_range) + 1), list(data['NumberOfAssignments']))
        plt.xticks(range(len(size_range) + 1), labels=['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-45', '>45'])
        plt.title("VERSES sequencing: Assignments of length N")
        plt.xlabel('Assignment length')
        plt.ylabel('Number of assignments')
        plt.savefig(f'{output_folder}{output_name}_aggregate_assignment_nums.png')
        plt.clf()
    
    return results


def analyze_batched_assignments_daily(batched_data_folder, optimized_unbatched_data, unoptimized_unbatched_data, iters):
    iters = str(iters).zfill(3)
    days = [pd.read_csv(fn) for fn in glob.glob(f'{batched_data_folder}/*.csv')]
    num_days = len(days)

    total_unoptimized_distance = total_optimized_distance = total_batched_distance = 0
    unoptimized_distances, optimized_distances, batched_distances = [], [], []

    dates = []

    for day in days:
        day.loc[:,('Date')] = pd.to_datetime(day.loc[:,('Date')]).copy()
        print(day['Date'])
        date = next(iter(day['Date']))
        dates.append(date)
        relevant_assignments = unoptimized_unbatched_data[unoptimized_unbatched_data['AssignmentEndTime'].dt.date.eq(date)]['AssignmentNumber'].copy()
        unbatched_results = optimized_unbatched_data[optimized_unbatched_data['AssignmentNumber'].isin(relevant_assignments)]
        unbatched_distances = unbatched_results['UnoptimizedDistance'].sum()/1000
        unbatched_optimized_distances = unbatched_results['OptimizedDistance'].sum()/1000
        batched_optimized_distances = day['BatchOptimizedDistance'].sum()/1000

        total_unoptimized_distance += unbatched_distances
        total_optimized_distance += unbatched_optimized_distances
        total_batched_distance += batched_optimized_distances

        unoptimized_distances.append(unbatched_distances)
        optimized_distances.append(unbatched_optimized_distances)
        batched_distances.append(batched_optimized_distances)

    plt.bar(range(num_days), unoptimized_distances, label='Unoptimized NRI assignment distances')
    plt.bar(range(num_days), optimized_distances, label='Optimized NRI assignment distances')
    plt.bar(range(num_days), batched_distances, label='Optimized VERSES assignment distances')
    plt.xticks([])
    plt.title("VERSES batching + sequencing")
    plt.xlabel('Date')
    plt.ylabel('Assignment length (km)')
    totals = {
        'total_unoptimized_distance': total_unoptimized_distance,
        'total_optimized_distance': total_optimized_distance,
        'total_batched_distance': total_batched_distance
    }
    totals = pd.DataFrame(totals, index=[0])
    os.makedirs(f"{batched_data_folder}/totals", exist_ok=True)
    totals.to_csv(f"{batched_data_folder}/totals/batching_analysis_totals_{iters}.csv")
    # plt.annotate(f"Total unoptimized distance: {total_unoptimized_distance:.2f}", (3,4))
    # plt.annotate(f"Total optimized distance: {total_optimized_distance:.2f} km", (3,7))
    # plt.annotate(f"Total batched/optimized distance: {total_batched_distance:.2f}", (3, 13))

    plt.legend()
    plt.savefig(f'{batched_data_folder}/batching_analysis_{iters}.png')
    plt.clf()


def analyze_batched_assignments(batched_data_folder, optimized_unbatched_data, unoptimized_unbatched_data, output_name, output_folder):
    output_folder = normalize_folder(output_folder)
    if output_name is None:
        output_name = output_folder
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    days = [pd.read_csv(fn) for fn in glob.glob(f'{batched_data_folder}/*.csv')]
    num_days = len(days)

    total_unoptimized_distance = total_optimized_distance = total_batched_distance = 0
    unoptimized_distances, optimized_distances, batched_distances = [], [], []

    dates = []

    for day in days:
        day.loc[:,('Date')] = pd.to_datetime(day.loc[:,('Date')]).copy()
        #print(day['Date'])
        date = next(iter(day['Date']))
        dates.append(date)
        relevant_assignments = unoptimized_unbatched_data[unoptimized_unbatched_data['AssignmentEndTime'].dt.date.eq(date)]['AssignmentNumber'].copy()
        unbatched_results = optimized_unbatched_data[optimized_unbatched_data['AssignmentNumber'].isin(relevant_assignments)]
        unbatched_distances = unbatched_results['UnoptimizedDistance'].sum()/1000
        unbatched_optimized_distances = unbatched_results['OptimizedDistance'].sum()/1000
        batched_optimized_distances = day['BatchOptimizedDistance'].sum()/1000

        total_unoptimized_distance += unbatched_distances
        total_optimized_distance += unbatched_optimized_distances
        total_batched_distance += batched_optimized_distances

        unoptimized_distances.append(unbatched_distances)
        optimized_distances.append(unbatched_optimized_distances)
        batched_distances.append(batched_optimized_distances)

    plt.bar(range(num_days), unoptimized_distances, label='Unoptimized NRI assignment distances')
    plt.bar(range(num_days), optimized_distances, label='Optimized NRI assignment distances')
    plt.bar(range(num_days), batched_distances, label='Optimized VERSES assignment distances')
    plt.xticks([])
    plt.title("VERSES batching + sequencing")
    plt.xlabel('Date')
    plt.ylabel('Assignment length (km)')
    totals = {
        'total_unoptimized_distance': total_unoptimized_distance,
        'total_optimized_distance': total_optimized_distance,
        'total_batched_distance': total_batched_distance
    }
    totals = pd.DataFrame(totals, index=[0])
    os.makedirs(f"{output_folder}", exist_ok=True)
    totals.to_csv(f"{output_folder}{output_name}_totals.csv")

    plt.legend()
    plt.savefig(f"{output_folder}{output_name}_analysis.png")
    plt.clf()
