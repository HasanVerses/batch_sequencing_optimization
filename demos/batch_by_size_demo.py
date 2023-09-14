import time

from opt.io.local import get_domain_graph, parse_assignment_input, get_assignments_of_len
from opt.api.snaking import compare_to_optimized_multi, optimize_picks
from multiprocessing import Pool



if __name__ == "__main__":

    DATA_PATH = 'data/dcs/nri/VA/VA_assignments.csv'
    data = parse_assignment_input(DATA_PATH)

    num_samples = 4

    MIN_SIZE = 5
    MAX_SIZE = 45
    BIN_SIZE = 5
    size_range = range(MIN_SIZE, MAX_SIZE, BIN_SIZE)

    VA = get_domain_graph("VA")

    lengthwise_bins = [get_assignments_of_len(
        data, [n, n + BIN_SIZE], randomize=True, num_assignments=num_samples, filter_by_graph=VA
    ) for n in size_range]

    pool = Pool()

    results = []
    times = []
    for idx, bin in enumerate(lengthwise_bins):
        print(f"Processing {num_samples} assignments of length {MIN_SIZE + (idx*BIN_SIZE)}...")
        results.append(compare_to_optimized_multi(VA, bin, n=4))
        # start = time.time()
        # results = optimize_picks(VA, bin[idx], n=4, max_retries = 0, random_seed=42, cumulative_distances=False, output_df=False)
        # end = time.time() - start
        # print("time: ", end, "results: ", results)
