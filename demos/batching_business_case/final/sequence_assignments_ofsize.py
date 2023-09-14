import pandas as pd
import glob
import time
import os
from opt.analysis import sequence_batched_tasks, analyze_batched_assignments
from opt.io.local import get_domain_graph, normalize_folder


if __name__ == "__main__":

    FULL_NRI_DATA_FN = 'data/LD_to_sequence.csv'
    unoptimized_unbatched_data = pd.read_csv(FULL_NRI_DATA_FN)
    unoptimized_unbatched_data.loc[:,('AssignmentEndTime')] = pd.to_datetime(unoptimized_unbatched_data.loc[:,('AssignmentEndTime')]).copy()

    OPTIMIZED_ASSIGNMENTS_FN = 'data/all_assignments_optimized_3.csv'
    optimized_unbatched_data = pd.read_csv(OPTIMIZED_ASSIGNMENTS_FN)    

    BATCHING_ASSIGNMENT_FOLDER = 'data/batches_of_size/'
 
    OUTPUT_FOLDER = normalize_folder('ofsize_sequenced')
    OUTPUT_NAME = 'size'


    size_range = range(10, 45, 5)

    for size in size_range:

        folder_name = normalize_folder(BATCHING_ASSIGNMENT_FOLDER + 'size' + str(size))
        output_name = OUTPUT_NAME + str(size)
        output_folder = normalize_folder(OUTPUT_FOLDER + OUTPUT_NAME + str(size))

        batch_days = [pd.read_csv(fn) for fn in glob.glob(f'{folder_name}/*.csv')]
        num_days = len(batch_days)

        LD = get_domain_graph("LD")

        analysis_interval = 30

        print("Optimizing assignments.")
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for idx, day in enumerate(batch_days):
            days_date = next(iter(day['Date']))
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
                result, seq = sequence_batched_tasks(batching_assignment, LD, n=1, date=days_date)
                if result is None:
                    print("NO results!")
                    time.sleep(4)
                else:
                    [results[k].append(v) for k, v in result.items()]

            df = pd.DataFrame(results)
            # Continuously save results to folder
            df.to_csv(f'{output_folder}{output_name}_{days_date}.csv')

            # Run analysis every `analysis_interval` steps
            if idx%analysis_interval == 0:
                analyze_batched_assignments(output_folder, optimized_unbatched_data, unoptimized_unbatched_data, output_name, output_folder + "totals")
