import glob
import pandas as pd
import matplotlib.pyplot as plt
from opt.analysis import convert_cols_to_datetime



ORIGINAL_DATA_PATH = 'data/to_sequence_2years.csv'
unoptimized_data = convert_cols_to_datetime(pd.read_csv(ORIGINAL_DATA_PATH))

assignments = unoptimized_data.groupby('AssignmentNumber')
total_time = 0
for no, data in assignments:
    endtimes = data['AssignmentEndTime']
    starttimes = data['AssignmentStartTime']

    endtime = next(iter(endtimes))
    starttime = next(iter(starttimes))
    total_time += pd.Timedelta(endtime - starttime).seconds

# total_time /= (60*60)
print("Duration (seconds): ", total_time)
minutes = total_time/60
print("Duration (minutes): ", minutes)
hours = minutes/60
print("Duration (hours): ", hours)

TOTAL_SECONDS = total_time
MILES_CONVERTER = 0.621371
WAGE = 25.87

PATH_SCHEMA = "ofsize_sequenced_2years/size{size}/totals/size{size}_totals.csv"

size_range = range(5, 45, 5)
unoptimized_distances, sequenced_distances, batched_distances = [], [], []


seq_sav, batched_seq_sav, batched_sav = [], [], []
seq_sav, batched_seq_sav, batched_sav = [], [], []
seq_time, batched_seq_time, batched_time = [], [], []
seq_money, batched_seq_money, batched_money = [], [], []


output_data = {
    "Sequenced VS unoptimized: km saved": [],
    "Batched VS sequenced: km saved": [],
    "Batched VS unoptimized: km saved": [],
    "Sequenced VS unoptimized: miles saved": [],
    "Batched VS sequenced: miles saved": [],
    "Batched VS unoptimized: miles saved": [],
    "Sequenced VS unoptimized: hours saved": [],
    "Batched VS sequenced: hours saved": [],
    "Batched VS unoptimized: hours saved": [],
    "Sequenced VS unoptimized: $ saved": [],
    "Batched VS sequenced: $ saved": [],
    "Batched VS unoptimized: $ saved": [],
    "Sequenced VS unoptimized: percent saved": [],
    "Batched VS sequenced: percent saved": [],
    "Batched VS unoptimized: percent saved": [],


}

for size in size_range:
    print("Size", size)
    data = pd.read_csv(PATH_SCHEMA.format(size=size))

    unopt_distance = data.loc[0,'total_unoptimized_distance']
    seq_distance = data.loc[0,'total_optimized_distance']
    bat_distance = data.loc[0,'total_batched_distance']

    unoptimized_distances.append(unopt_distance)
    sequenced_distances.append(seq_distance)
    batched_distances.append(bat_distance)

    sequenced_gap = unopt_distance - seq_distance
    batched_sequenced_gap = seq_distance - bat_distance
    batched_gap = unopt_distance - bat_distance

    output_data["Sequenced VS unoptimized: km saved"].append(sequenced_gap)
    output_data["Batched VS sequenced: km saved"].append(batched_sequenced_gap)
    output_data["Batched VS unoptimized: km saved"].append(batched_gap)
    output_data["Sequenced VS unoptimized: miles saved"].append(sequenced_gap*MILES_CONVERTER)
    output_data["Batched VS sequenced: miles saved"].append(batched_sequenced_gap*MILES_CONVERTER)
    output_data["Batched VS unoptimized: miles saved"].append(batched_gap*MILES_CONVERTER)

    sequenced_savings = sequenced_gap/unopt_distance
    batched_sequenced_savings = batched_sequenced_gap/seq_distance
    output_data["Batched VS sequenced: percent saved"].append(batched_sequenced_savings)

    batched_savings = batched_gap/unopt_distance
    output_data["Batched VS unoptimized: percent saved"].append(batched_savings)

    # seq_sav.append(sequenced_savings)
    # batched_seq_sav.append(batched_sequenced_savings)
    # batched_sav.append(batched_savings)

    hours_saved_seq = TOTAL_SECONDS*sequenced_savings/(60*60)
    hours_saved_seq_bat = TOTAL_SECONDS*batched_sequenced_savings/(60*60)
    hours_saved_bat = TOTAL_SECONDS*batched_savings/(60*60)

    dollars_saved_seq = WAGE*hours_saved_seq
    dollars_saved_seq_bat = WAGE*hours_saved_seq_bat
    dollars_saved_bat = WAGE*hours_saved_bat

    print("Savings (sequenced):", hours_saved_seq, "hours")
    print("Cost savings: ", dollars_saved_seq)

    print("Savings (batched VS sequenced):", hours_saved_seq_bat, "hours")
    print("Cost savings: ", dollars_saved_seq_bat)

    print("Savings (batched):", hours_saved_bat, "hours")
    print("Cost savings: ", dollars_saved_bat)

    seq_time.append(hours_saved_seq)
    batched_seq_time.append(hours_saved_seq_bat)
    batched_time.append(hours_saved_bat)

    seq_money.append(dollars_saved_seq)
    batched_seq_money.append(dollars_saved_seq_bat)
    batched_money.append(dollars_saved_bat)

savings_df = pd.DataFrame(
    {"Size": list(size_range),
    "Batched vs unoptimized: km saved": output_data["Batched VS unoptimized: km saved"],
    "Batched vs sequenced: km saved": output_data["Batched VS sequenced: km saved"],
    "Batched vs unoptimized: miles saved": output_data["Batched VS unoptimized: miles saved"],
    "Batched vs sequenced: miles saved": output_data["Batched VS sequenced: miles saved"],
    "Batched vs unoptimized: hours saved": batched_time,
    "Batched vs sequenced: hours saved": batched_seq_time, 
    "Batched vs unoptimized: $ saved": batched_money,
    "Batched vs sequenced: $ saved": batched_seq_money,
    "Batched vs unoptimized: percent savings": output_data["Batched VS unoptimized: percent saved"],
    "Batched vs sequenced: percent saved": output_data["Batched VS sequenced: percent saved"]
    }
)
savings_df.to_csv("LD_batching_savings_2years.csv")

plt.plot(size_range, unoptimized_distances, label='Unoptimized NRI assignments')
plt.plot(size_range, sequenced_distances, label='Sequenced NRI assignments')
plt.plot(size_range, batched_distances, label='Sequenced VERSES assignments')
plt.title("VERSES batching + sequencing")
plt.xlabel('Target assignment size')
plt.ylabel('Total distance (km)')
plt.legend()

plt.savefig("batching_results_per_size_2years.png")
plt.clf()


# Variances

PATH_SCHEMA = "ofsize_sequenced_2years/size{size}/*.csv" 
means = []
variances = []

for size in size_range:
    days = [pd.read_csv(fn) for fn in glob.glob(PATH_SCHEMA.format(size=size))]

    all_data = pd.concat(days, ignore_index=True)
    means.append(all_data['BatchAssignmentLength'].mean())
    variances.append(all_data['BatchAssignmentLength'].var())
    print(f"variance size {size}: {all_data['BatchAssignmentLength'].var()}")

plt.title("Target VS generated batch sizes")
plt.scatter(size_range, size_range, label='Target sizes', color='orange')
plt.errorbar(size_range, means, yerr=variances, label='Mean generated batch sizes')
plt.legend(loc='upper left')
plt.savefig("batching_results_batch_sizes_2years.png")
