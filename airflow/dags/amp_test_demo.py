"""AMP Test Demo."""
from datetime import datetime

import numpy as np
import pandas as pd
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator

import sys
sys.path.append('/usr/local/code')
import public_timeseries_testing_util as amp_pd_peptide
env = amp_pd_peptide.make_env()
iter_test = env.iter_test()

def _apm_demo(**context):

    counter = 0
    # The API will deliver four dataframes in this specific order:
    for (test, test_peptides, test_proteins, sample_submission) in iter_test:
        if counter == 0:
            print(test.head())
            print(test_peptides.head())
            print(test_proteins.head())
            print(sample_submission.head())
            sample_submission['rating'] = 0
        env.predict(sample_submission)
        counter += 1

    return

default_args = {'owner': 'hernan', 'retries': 0, 'start_date': datetime(2022, 8, 14)}
with DAG('amp_test_demo', default_args=default_args, schedule_interval=None) as dag:
    generate_demo_submission = PythonOperator(
        task_id='generate_demo_submission',
        python_callable=_apm_demo,
        op_args='',
        provide_context=True,
    )

    generate_demo_submission
