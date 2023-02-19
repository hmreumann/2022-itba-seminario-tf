# ITBA 2022 - Seminario - Trabajo Final
## Start the local environment
./control-env.sh start

## Introduction
We used a Kaggle [AMP-Parkinson's Disease Progression Prediction](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/overview) competition to apply the tools seen during the competition.

## Airflow
We chose Airflow as the tool to dispatch the different jobs in order to build a final model. It provides an easy way to show the general process, and at the same time the possibility to dive into the details of each task if needed.

After starting the environment, run the [Airflow Dashobard](http://localhost:9090)

## AMP Test
Trigger amp_test_demo DAG, and look for the Log to see the results of the AMP API.

Ref.: [Basic API Demo](https://www.kaggle.com/code/sohier/basic-api-demo)
