import numpy as np
import pandas as pd
import time
import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from opt.io.local import get_domain_graph
from opt.api.snaking import optimize_picks



def run_test_batching(assignment_data, graph, n=1, algorithm='annealing', save_results=False, date=None, delivery_zone_name='Delivery zone'):
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


def analyze_assignments(batched_data_folder, optimized_unbatched_data, unoptimized_unbatched_data, iters):
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


if __name__ == "__main__":

    BATCHING_ASSIGNMENT_FOLDER = 'data/batching_test_3/verses_batches'
    ORIGINAL_ASSIGNMENTS_FN = 'data/all_assignments_optimized_3.csv'
    FULL_NRI_DATA_FN = 'data/LD_to_sequence.csv'
    batch_days = [pd.read_csv(fn) for fn in glob.glob(f'{BATCHING_ASSIGNMENT_FOLDER}/*.csv')]
    num_days = len(batch_days)

    optimized_unbatched_data = pd.read_csv(ORIGINAL_ASSIGNMENTS_FN)    
    original_unbatched_data = pd.read_csv(FULL_NRI_DATA_FN)
    original_unbatched_data.loc[:,('AssignmentEndTime')] = pd.to_datetime(original_unbatched_data.loc[:,('AssignmentEndTime')]).copy()

    LD = get_domain_graph("LD")

    ###---------Run tests----------###

    analysis_interval = 30

    print("Optimizing assignments.")

    OUTPUT_FOLDER = 'batching_test_3_results'
    os.makedirs(OUTPUT_FOLDER)

    for idx, day in enumerate(batch_days):
        days_date = next(iter(day['AssignmentStartTime']))
        print(f"Processing assignments for {days_date}")
        results = {
            'Date': [],
            'BatchAssignmentNumber': [],
            'BatchAssignmentLength': [],
            'BatchOptimizedSequence': [],
            'BatchOptimizedDistance': [],
            'BatchExecutionTime': [],
            'BatchNumParallelRuns': [],
            'BatchAlgorithm': [] 
        }

        by_batching_assignment = day.groupby(['BatchingAssignmentNumber'])

        for batching_no, batching_assignment in by_batching_assignment:
            result, seq = run_test_batching(batching_assignment, LD, n=1, date=days_date)
            if result is None:
                print("NO results!")
                time.sleep(4)
            else:
                [results[k].append(v) for k, v in result.items()]

        df = pd.DataFrame(results)
        # Continuously save results to folder
        day.loc[:,('AssignmentStartTime')] = pd.to_datetime(day.loc[:,('AssignmentStartTime')])
        df.to_csv(f'{OUTPUT_FOLDER}/LD_sequenced_batched_assignments_{days_date}.csv')

        # Run analysis every `analysis_interval` steps
        if idx%analysis_interval == 0:
            analyze_assignments(OUTPUT_FOLDER, optimized_unbatched_data, original_unbatched_data, idx)
