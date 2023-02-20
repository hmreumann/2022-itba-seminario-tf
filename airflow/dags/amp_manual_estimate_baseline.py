"""AMP Manual Estimate Baseline."""
# https://www.kaggle.com/code/danielpeshkov/manual-estimate-baseline
from datetime import datetime

import numpy as np
import pandas as pd
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator

# Smape
def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))

    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)

    pos_ind = dem != 0
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]

    return 100 * np.mean(smap)

# Train
train = pd.read_csv('/usr/local/dataset/train_clinical_data.csv')


estimates = {}
months = train.visit_month.unique()
targets = ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']
for m in months:
    for target in targets:
        t = train[train.visit_month==m][f'{target}'].dropna().values
        if len(t) >= 200:
            s = []
            best_threshold = 0
            best_smape = 200
            for i in np.arange(0, 30, 0.1):
                score = smape(t, np.array([i for _ in range(len(t))]))
                s.append(score)
                if score < best_smape:
                    best_smape = score
                    best_threshold = i
        else:
            best_threshold = np.median(t)
        estimates[(m, target)] = best_threshold

for i in range(sorted(months)[-1]+1):
    for target in targets:
        if (i, target) not in estimates:
            estimates[(i, target)] = estimates[(i-1, target)]

# We import the time series testing tool
import sys
sys.path.append('/usr/local/code')
import public_timeseries_testing_util as amp_pd_peptide
env = amp_pd_peptide.make_env()
iter_test = env.iter_test()

def _apm_manual_estimate_baseline(**context):

    # The API will deliver four dataframes in this specific order:
    for (test, test_peptides, test_proteins, sample_submission) in iter_test:
        # This maps the correct value estimate to each line in sample_submission
        targets = sample_submission.prediction_id.str.split('_').apply(lambda x: (int(x[1]) + int(x[5]), '_'.join(x[2:4])))
        sample_submission['rating'] = targets.map(estimates)

        print(sample_submission.head())

        env.predict(sample_submission)

    return

default_args = {'owner': 'hernan', 'retries': 0, 'start_date': datetime(2022, 8, 14)}
with DAG('amp_manual_estimate_baseline', default_args=default_args, schedule_interval=None) as dag:
    manual_estimate_baseline = PythonOperator(
        task_id='manual_estimate_baseline',
        python_callable=_apm_manual_estimate_baseline,
        op_args='',
        provide_context=True,
    )

    manual_estimate_baseline
