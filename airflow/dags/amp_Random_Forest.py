"""AMP Random Forest."""
from datetime import datetime

import numpy as np
import pandas as pd
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator



"Smape"
def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    pos_ind = (y_true!=0)|(y_pred!=0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    
    return 100 * np.mean(smap)


"Protein Data"
proteins = pd.read_csv('/usr/local/dataset/train_proteins.csv')

"Test Protein Data"
proteins_test = pd.read_csv('/usr/local/dataset/example_test_files/test_proteins.csv')

"Peptides Data"
peptides = pd.read_csv('/usr/local/dataset/train_peptides.csv')

"Test Protein Data"
peptides_test = pd.read_csv('/usr/local/dataset/example_test_files/test_peptides.csv')

"Clinical Data"
clinical = pd.read_csv('/usr/local/dataset/train_clinical_data.csv')

"Test Clinical Data"
test = pd.read_csv('/usr/local/dataset/example_test_files/test.csv')

"Group Proteins"
proteins.groupby('visit_id').agg({'UniProt':'nunique','patient_id':'count','NPX':['min','max','mean','std']}).reset_index()

"Group Peptides"
peptides.groupby('visit_id').agg({'UniProt':'nunique','patient_id':'count','Peptide':'nunique','PeptideAbundance': ['min','max','mean','std']}).reset_index()


"Dataset para Training (primer mes)"
df_0 = clinical[(clinical.visit_month == 0)][['visit_id','updrs_1']]

"Feature Engineering (proteins features)"
proteins_npx_ft = proteins.groupby('visit_id').agg(NPX_min=('NPX','min'), NPX_max=('NPX','max'), NPX_mean=('NPX','mean'), NPX_std=('NPX','std')).reset_index()


df_proteins = pd.merge(proteins, df_0, on = 'visit_id', how = 'inner').reset_index()

proteins_Uniprot_updrs = df_proteins.groupby('UniProt').agg(updrs_1_sum = ('updrs_1','mean')).reset_index()

df_proteins = pd.merge(proteins, proteins_Uniprot_updrs, on = 'UniProt', how = 'left')

proteins_UniProt_ft = df_proteins.groupby('visit_id').agg(proteins_updrs_1_min=('updrs_1_sum','min'), proteins_updrs_1_max=('updrs_1_sum','max'),proteins_updrs_1_mean=('updrs_1_sum','mean'), proteins_updrs_1_std=('updrs_1_sum','std')).reset_index()

"Feature Engineering (peptides features)"

peptides_PeptideAbundance_ft = peptides.groupby('visit_id').agg(Abe_min=('PeptideAbundance','min'), Abe_max=('PeptideAbundance','max'),Abe_mean=('PeptideAbundance','mean'), Abe_std=('PeptideAbundance','std')).reset_index()


df_peptides = pd.merge(peptides, df_0, on = 'visit_id', how = 'inner').reset_index()
peptides_PeptideAbundance_updrs = df_peptides.groupby('Peptide').agg(updrs_1_sum = ('updrs_1','mean')).reset_index()

"Todos los df`s juntos"

df_0_1 = clinical[(clinical.visit_month == 3)][['visit_id','updrs_1']]
df_0_2 = clinical[(clinical.visit_month == 3)][['visit_id','updrs_2']]
df_0_3 = clinical[(clinical.visit_month == 3)][['visit_id','updrs_3']]
df_0_4 = clinical[(clinical.visit_month == 3)][['visit_id','updrs_4']]

df_proteins = pd.merge(proteins, df_0_1, on = 'visit_id', how = 'inner').reset_index()
proteins_Uniprot_updrs1 = df_proteins.groupby('UniProt').agg(updrs_1_sum = ('updrs_1','mean')).reset_index()

df_proteins = pd.merge(proteins, df_0_2, on = 'visit_id', how = 'inner').reset_index()
proteins_Uniprot_updrs2 = df_proteins.groupby('UniProt').agg(updrs_1_sum = ('updrs_2','mean')).reset_index()

df_proteins = pd.merge(proteins, df_0_3, on = 'visit_id', how = 'inner').reset_index()
proteins_Uniprot_updrs3 = df_proteins.groupby('UniProt').agg(updrs_1_sum = ('updrs_3','mean')).reset_index()

df_proteins = pd.merge(proteins, df_0_4, on = 'visit_id', how = 'inner').reset_index()
proteins_Uniprot_updrs4 = df_proteins.groupby('UniProt').agg(updrs_1_sum = ('updrs_4','mean')).reset_index()

df_peptides = pd.merge(peptides, df_0_1, on = 'visit_id', how = 'inner').reset_index()
peptides_PeptideAbundance_updrs1 = df_peptides.groupby('Peptide').agg(updrs_1_sum = ('updrs_1','mean')).reset_index()

df_peptides = pd.merge(peptides, df_0_2, on = 'visit_id', how = 'inner').reset_index()
peptides_PeptideAbundance_updrs2 = df_peptides.groupby('Peptide').agg(updrs_1_sum = ('updrs_2','mean')).reset_index()

df_peptides = pd.merge(peptides, df_0_3, on = 'visit_id', how = 'inner').reset_index()
peptides_PeptideAbundance_updrs3 = df_peptides.groupby('Peptide').agg(updrs_1_sum = ('updrs_3','mean')).reset_index()

df_peptides = pd.merge(peptides, df_0_4, on = 'visit_id', how = 'inner').reset_index()
peptides_PeptideAbundance_updrs4 = df_peptides.groupby('Peptide').agg(updrs_1_sum = ('updrs_4','mean')).reset_index()

df_proteins_fts = [proteins_Uniprot_updrs1, proteins_Uniprot_updrs2, proteins_Uniprot_updrs3, proteins_Uniprot_updrs4]
df_peptides_fts = [peptides_PeptideAbundance_updrs1, peptides_PeptideAbundance_updrs2, peptides_PeptideAbundance_updrs3, peptides_PeptideAbundance_updrs4]
df_lst = [df_0_1, df_0_2, df_0_3, df_0_4]

def features(df, proteins, peptides, classes):
    proteins_npx_ft = proteins.groupby('visit_id').agg(NPX_min=('NPX','min'), NPX_max=('NPX','max'), NPX_mean=('NPX','mean'), NPX_std=('NPX','std')).reset_index()
    peptides_PeptideAbundance_ft = peptides.groupby('visit_id').agg(Abe_min=('PeptideAbundance','min'), Abe_max=('PeptideAbundance','max'),Abe_mean=('PeptideAbundance','mean'), Abe_std=('PeptideAbundance','std')).reset_index()

    df_proteins = pd.merge(proteins, df_proteins_fts[classes], on = 'UniProt', how = 'left')
    proteins_UniProt_ft = df_proteins.groupby('visit_id').agg(proteins_updrs_1_min=('updrs_1_sum','min'), proteins_updrs_1_max=('updrs_1_sum','max'),proteins_updrs_1_mean=('updrs_1_sum','mean'), proteins_updrs_1_std=('updrs_1_sum','std')).reset_index()
    df_peptides = pd.merge(peptides, df_peptides_fts[classes], on = 'Peptide', how = 'left')
    peptides_ft = df_peptides.groupby('visit_id').agg(peptides_updrs_1_min=('updrs_1_sum','min'), peptides_updrs_1_max=('updrs_1_sum','max'),peptides_updrs_1_mean=('updrs_1_sum','mean'), peptides_updrs_1_std=('updrs_1_sum','std')).reset_index()

    df = pd.merge(df, proteins_npx_ft, on = 'visit_id', how = 'left')
    df = pd.merge(df, peptides_PeptideAbundance_ft, on = 'visit_id', how = 'left')
    df = pd.merge(df, proteins_UniProt_ft, on = 'visit_id', how = 'left')
    df = pd.merge(df, peptides_ft, on = 'visit_id', how = 'left')
    df = df.fillna(df.mean())
    return df

train_0 = features(df_0_1, proteins, peptides, 0)

"Training"
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error, make_scorer

from sklearn.metrics import confusion_matrix,precision_score,recall_score,classification_report, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

model = {}
mms = MinMaxScaler()
n_estimators = [5,20,50,100] # number of trees in the random forest
max_features = ['auto', 'sqrt'] # number of features in consideration at every split
max_depth = [int(x) for x in np.linspace(10, 120, num = 12)] # maximum number of levels allowed in each decision tree
min_samples_split = [2, 6, 10] # minimum sample number to split a node
min_samples_leaf = [1, 3, 4] # minimum sample number that can be stored in a leaf node
bootstrap = [True, False]

for i in range(2):
    print('--------------------------------------------------------')
    print('Model {0}'.format(i + 1))
    train_0 = features(df_lst[i], proteins, peptides, i)
    scale_col = ['NPX_min','NPX_max','NPX_mean','NPX_std', 'Abe_min', 'Abe_max', 'Abe_mean', 'Abe_std']
    train_0[scale_col] = mms.fit_transform(train_0[scale_col])
    
    rfc = RandomForestRegressor()
    forest_params = [{'n_estimators': n_estimators,

    'max_features': max_features,

    'max_depth': max_depth,

    'min_samples_split': min_samples_split,

    'min_samples_leaf': min_samples_leaf,

    'bootstrap': bootstrap}]
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    clf = RandomizedSearchCV(rfc, forest_params, cv = cv, scoring=make_scorer(smape), verbose = -1)
    
    X = train_0.drop(columns = ['visit_id','updrs_{0}'.format(i + 1)], axis = 1)
    y = train_0['updrs_{0}'.format(i + 1)]
    clf.fit(X, y)

    print(clf.best_params_)

    print(clf.best_score_)
    print('Train smape:',smape(train_0['updrs_{0}'.format(i + 1)], clf.predict(train_0.drop(columns = ['visit_id','updrs_{0}'.format(i + 1)], axis = 1))))
    model[i] = clf

updrs_3_pred = {}
up3 = clinical[['visit_month','updrs_3']].drop_duplicates(['visit_month','updrs_3'])
updrs_3_pred = dict(zip(up3.visit_month, up3.updrs_3))
updrs_3_pred

"Inferencia"
import amp_pd_peptide
env = amp_pd_peptide.make_env()
iter_test = env.iter_test()

def map_test(x):
    updrs = x.split('_')[2] + '_' + x.split('_')[3]
    month = int(x.split('_plus_')[1].split('_')[0])
    visit_id = x.split('_')[0] + '_' + x.split('_')[1]
    # set all predictions 0 where updrs equals 'updrs_4'
    if updrs=='updrs_3':
        rating = updrs_3_pred[month]
    elif updrs=='updrs_4':
        rating = 0
    elif updrs =='updrs_1':
        rating = df[df.visit_id == visit_id]['pred0'].values[0]
    else:
        rating = df[df.visit_id == visit_id]['pred1'].values[0]
    return rating

counter = 0
for (test, test_peptides, test_proteins, sample_submission) in iter_test:
    df = test[['visit_id']].drop_duplicates('visit_id')
    pred_0 = features(df[['visit_id']], test_proteins, test_peptides, 0)
    scale_col = ['NPX_min','NPX_max','NPX_mean','NPX_std', 'Abe_min', 'Abe_max', 'Abe_mean', 'Abe_std']
    pred_0[scale_col] = mms.fit_transform(pred_0[scale_col])
    pred_0 = model[0].predict(pred_0.drop(columns = ['visit_id'], axis = 1))
    df['pred0'] = pred_0
    
    pred_1 = features(df[['visit_id']], test_proteins, test_peptides, 1)
    scale_col = ['NPX_min','NPX_max','NPX_mean','NPX_std', 'Abe_min', 'Abe_max', 'Abe_mean', 'Abe_std']
    pred_1[scale_col] = mms.fit_transform(pred_1[scale_col])
    pred_1 = model[1].predict(pred_1.drop(columns = ['visit_id'], axis = 1))
    df['pred1'] = pred_1
    
    sample_submission['rating'] = sample_submission['prediction_id'].apply(map_test)
    env.predict(sample_submission)
    
    if counter == 0:
        display(test)
        display(sample_submission)
        
    counter += 1

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


default_args = {'owner': 'justo', 'retries': 0, 'start_date': datetime(2022, 8, 14)}
with DAG('amp_Random_Forest', default_args=default_args, schedule_interval=None) as dag:
    generate_demo_submission = PythonOperator(
        task_id='RF_submission',
        python_callable=_apm_demo,
        op_args='',
        provide_context=True,
    )

    generate_demo_submission

