import math
import pandas as pd

data = pd.read_csv('crowdflower_data_transform/lon2_agg.csv')

# {worker_id: [N correctly tagged papers, total tagged papers]}
workers_summary = {}
data_summary_temp = []
# iterate over data rows
# compute statistic of answers
for _, row in data.iterrows():
    worker_id = int(row['worker_id'])
    w_ans = row.values[2:]
    n_correct_tagged = 0
    n_total_tagged = 0
    for vote in w_ans:
        if vote == 3:
            n_correct_tagged += 1
            n_total_tagged += 1
        elif not math.isnan(vote):
            n_total_tagged += 1
    workers_summary.update({worker_id: [n_correct_tagged, n_total_tagged]})
    data_summary_temp.append([n_correct_tagged, n_total_tagged])

data_summary = pd.DataFrame(data=data_summary_temp, columns=['n_correct', 'n_total'])
data_summary = data_summary.groupby(['n_correct', 'n_total']).size().reset_index(name="Time")
