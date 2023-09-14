import pandas as pd
from opt.analysis  import convert_cols_to_datetime



FULL_NRI_DATA_FN = 'data/LD_to_sequence.csv'
uud = convert_cols_to_datetime(pd.read_csv(FULL_NRI_DATA_FN))
assignments = uud.groupby('AssignmentNumber')
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