import pandas as pd
import glob
import time
import os
import datetime

from tqdm import tqdm
from opt.analysis import sequence_batched_tasks, analyze_batched_assignments, run_sequencing_test, convert_cols_to_datetime
from opt.io.local import get_domain_graph, normalize_folder
from opt.model.sequence import Sequence


if __name__ == "__main__":

    print("Loading graph...")
    LD = get_domain_graph("LD")

    TEST_ID = "2years"

    print("Loading data...")
    FULL_NRI_DATA_FN = 'data/to_sequence_all_ecomm.csv'
    unoptimized_unbatched_data = pd.read_csv(FULL_NRI_DATA_FN)
    unoptimized_unbatched_data.loc[:,('AssignmentEndTime')] = pd.to_datetime(unoptimized_unbatched_data.loc[:,('AssignmentEndTime')]).copy()

    #Grab first month of data only
    unoptimized_unbatched_data = unoptimized_unbatched_data[unoptimized_unbatched_data['AssignmentEndTime'].dt.date.le(datetime.date(2020, 2, 1))]
    unoptimized_unbatched_data.to_csv('to_sequence_2years.csv')
    # unoptimized_unbatched_data = convert_cols_to_datetime(pd.read_csv('data/to_sequence_2years.csv'))
    by_date_and_assignment = unoptimized_unbatched_data.groupby(
        [unoptimized_unbatched_data['AssignmentEndTime'].dt.date, unoptimized_unbatched_data['AssignmentNumber']])

    #First, sequence all assignments # NOTE: DONE
    results = {
        'AssignmentNumber': [],
        'AssignmentLength': [],
        'UnoptimizedSequence': [],
        'OptimizedSequence': [],
        'UnoptimizedDistance': [],
        'OptimizedDistance': [],
        'DistanceSavings': [],
        'SavingsPercent': [],
        'ExecutionTime': [],
        'NumParallelRuns': [],
        'Algorithm': []
    }
    folder_name = f"sequenced_{TEST_ID}"
    filename = f"LD_sequenced_{TEST_ID}.csv"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for idx, group in tqdm(by_date_and_assignment):
        date, assignment_no = idx

        print(f"Processing assignment {idx}")

        result, seq, opt_seq = run_sequencing_test(group, LD, n=4)
        seq = Sequence(seq, LD)
        opt_seq = Sequence(opt_seq, LD)

        if result is None:
            print("NO results!")
            time.sleep(4)
        else:
            [results[k].append(v) for k, v in result.items()]

        df = pd.DataFrame(results)
        
        # Continuously save results to folder
        df.to_csv(f'{folder_name}/{filename}')

    # Now sequence the batched tasks and compare

    OPTIMIZED_ASSIGNMENTS_FN = f'data/{folder_name}/{filename}'
    optimized_unbatched_data = pd.read_csv(OPTIMIZED_ASSIGNMENTS_FN)    

    BATCHING_ASSIGNMENT_FOLDER = 'data/batches_of_size_2years/'
 
    OUTPUT_FOLDER = normalize_folder('ofsize_sequenced_2years')
    OUTPUT_NAME = 'size'

    size_range = range(5, 45, 5)

    for size in size_range:

        folder_name = normalize_folder(BATCHING_ASSIGNMENT_FOLDER + 'size' + str(size))
        output_name = OUTPUT_NAME + str(size)
        output_folder = normalize_folder(OUTPUT_FOLDER + OUTPUT_NAME + str(size))

        batch_days = [pd.read_csv(fn) for fn in glob.glob(f'{folder_name}/*.csv')]
        num_days = len(batch_days)

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
                result, seq = sequence_batched_tasks(batching_assignment, LD, n=4, date=days_date)
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
