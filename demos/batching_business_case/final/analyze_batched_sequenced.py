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


    size_range = range(5, 45, 5)
    LD = get_domain_graph("LD")

    for size in size_range:

        folder_name = normalize_folder(BATCHING_ASSIGNMENT_FOLDER + 'size' + str(size))
        output_name = OUTPUT_NAME + str(size)
        output_folder = normalize_folder(OUTPUT_FOLDER + OUTPUT_NAME + str(size))

        analyze_batched_assignments(output_folder, optimized_unbatched_data, unoptimized_unbatched_data, output_name, output_folder + "totals")
