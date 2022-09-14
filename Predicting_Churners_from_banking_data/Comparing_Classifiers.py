from operator import itemgetter

import pandas as pd 
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

import data_preprocessing

from copy import deepcopy

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SequentialFeatureSelector

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# imblearn libraries
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


# Pandas display options
pd.set_option('display.max_columns', 130)
pio.renderers = 'plotly'

#Function declarations:

def MinMaxNorm(df):
    return (df - df.min()) / (df.max() - df.min())

# Read files to be processed for the training dataset:
# read .csv files for train dataset
train_1 = pd.read_csv(r'data\\train_month_1.csv')
train_2 = pd.read_csv(r'data\\train_month_2.csv')
train_3 = pd.read_csv(r'data\\train_month_3_with_target.csv')

# Clean all 3 train dataframes, but keep the client_id column to match between customers
train_1_clean = data_preprocessing.remove_na_map_categorical(train_1)
train_2_clean = data_preprocessing.remove_na_map_categorical(train_2)
train_3_clean = data_preprocessing.remove_na_map_categorical(train_3)

# Bin balance and cap values
train_1_clean = data_preprocessing.bin_bal_cap(df=train_1_clean, nbins=10)
train_2_clean = data_preprocessing.bin_bal_cap(df=train_2_clean, nbins=10)
train_3_clean = data_preprocessing.bin_bal_cap(df=train_3_clean, nbins=10)

# Bin customer birth date column
train_1_clean = data_preprocessing.bin_birthdate(df=train_1_clean)
train_2_clean = data_preprocessing.bin_birthdate(df=train_2_clean)
train_3_clean = data_preprocessing.bin_birthdate(df=train_3_clean)

# Bin customer_since_all and customer_since_bank columns
train_1_clean = data_preprocessing.bin_customer_time(df=train_1_clean)
train_2_clean = data_preprocessing.bin_customer_time(df=train_2_clean)
train_3_clean = data_preprocessing.bin_customer_time(df=train_3_clean)

train_1_clean.loc[:, 'customer_postal_code'] = data_preprocessing.bin_postcode(train_1_clean['customer_postal_code'])
train_2_clean.loc[:, 'customer_postal_code'] = data_preprocessing.bin_postcode(train_2_clean['customer_postal_code'])
train_3_clean.loc[:, 'customer_postal_code'] = data_preprocessing.bin_postcode(train_3_clean['customer_postal_code'])

train_1_clean.set_index(keys=['client_id'], inplace=True)
train_2_clean.set_index(keys=['client_id'], inplace=True)
train_3_clean.set_index(keys=['client_id'], inplace=True)

# Read files to be processed for the test dataset
# Read .csv files for test dataset
test_1 = pd.read_csv('data\\test_month_1.csv')
test_2 = pd.read_csv('data\\test_month_2.csv')
test_3 = pd.read_csv('data\\test_month_3.csv')

# Clean all 3 train dataframes, but keep the client_id column to match between customers
test_1_clean = data_preprocessing.remove_na_map_categorical(test_1)
test_2_clean = data_preprocessing.remove_na_map_categorical(test_2)
test_3_clean = data_preprocessing.remove_na_map_categorical(test_3)

# Bin balance and cap values
test_1_clean = data_preprocessing.bin_bal_cap(df=test_1_clean, nbins=10)
test_2_clean = data_preprocessing.bin_bal_cap(df=test_2_clean, nbins=10)
test_3_clean = data_preprocessing.bin_bal_cap(df=test_3_clean, nbins=10)

# Bin customer birth date column
test_1_clean = data_preprocessing.bin_birthdate(df=test_1_clean)
test_2_clean = data_preprocessing.bin_birthdate(df=test_2_clean)
test_3_clean = data_preprocessing.bin_birthdate(df=test_3_clean)

# Bin customer_since_all and customer_since_bank columns
test_1_clean = data_preprocessing.bin_customer_time(df=test_1_clean)
test_2_clean = data_preprocessing.bin_customer_time(df=test_2_clean)
test_3_clean = data_preprocessing.bin_customer_time(df=test_3_clean)

test_1_clean.loc[:, 'customer_postal_code'] = data_preprocessing.bin_postcode(test_1_clean['customer_postal_code'])
test_2_clean.loc[:, 'customer_postal_code'] = data_preprocessing.bin_postcode(test_2_clean['customer_postal_code'])
test_3_clean.loc[:, 'customer_postal_code'] = data_preprocessing.bin_postcode(test_3_clean['customer_postal_code'])

test_1_clean.set_index(keys=['client_id'], inplace=True)
test_2_clean.set_index(keys=['client_id'], inplace=True)
test_3_clean.set_index(keys=['client_id'], inplace=True)

# Calculate total number of products for each month
# TRAIN dataset
train_1_clean.insert(loc=0, column='num_products_active',
     value=train_1_clean.apply(lambda x: x['homebanking_active':'has_current_account_starter'].sum(), axis=1))

train_2_clean.insert(loc=0, column='num_products_active',
     value=train_2_clean.apply(lambda x: x['homebanking_active':'has_current_account_starter'].sum(), axis=1))

train_3_clean.insert(loc=0, column='num_products_active',
     value=train_3_clean.apply(lambda x: x['homebanking_active':'has_current_account_starter'].sum(), axis=1))


# TEST dataset
test_1_clean.insert(loc=0, column='num_products_active',
     value=test_1_clean.apply(lambda x: x['homebanking_active':'has_current_account_starter'].sum(), axis=1))

test_2_clean.insert(loc=0, column='num_products_active',
     value=test_2_clean.apply(lambda x: x['homebanking_active':'has_current_account_starter'].sum(), axis=1))

test_3_clean.insert(loc=0, column='num_products_active',
     value=test_3_clean.apply(lambda x: x['homebanking_active':'has_current_account_starter'].sum(), axis=1))


# Cast all values to integers
train_1_clean = train_1_clean.apply(lambda x: x.astype(int))
train_2_clean = train_2_clean.apply(lambda x: x.astype(int))
train_3_clean = train_3_clean.apply(lambda x: x.astype(int))

test_1_clean = test_1_clean.apply(lambda x: x.astype(int))
test_2_clean = test_2_clean.apply(lambda x: x.astype(int))
test_3_clean = test_3_clean.apply(lambda x: x.astype(int))

# Add suffixes and merge all columns for all month for train and test dataset
train_1_clean = train_1_clean.add_suffix('_month1')
train_2_clean = train_2_clean.add_suffix('_month2')
train_3_clean = train_3_clean.add_suffix('_month3')

test_1_clean = test_1_clean.add_suffix('_month1')
test_2_clean = test_2_clean.add_suffix('_month2')
test_3_clean = test_3_clean.add_suffix('_month3')

train_dataset = pd.concat([train_1_clean, train_2_clean, train_3_clean], axis=1)
test_dataset = pd.concat([test_1_clean, test_2_clean, test_3_clean], axis=1)


# Separate features from targets and name accordingly
X_train_original = train_dataset[sorted(list(set(train_dataset.columns) - {'target_month3'}))]
y_train_original = train_dataset['target_month3']

X_test_original = test_dataset[sorted(test_dataset.columns)]

# Classifiers to be tested
classifiers = [
    ('Adaboost', AdaBoostClassifier(n_estimators=100, learning_rate=0.6, random_state=256)),
    ('SVM', svm.SVC(C=1, random_state=256)),
    ('LogisticRegression', LogisticRegression(n_jobs=-1, random_state=256)),
    ('RandomForest', RandomForestClassifier(n_estimators=100, criterion='gini', n_jobs=-1, random_state=256)),
    ('GaussianNB', GaussianNB()),
    ('DecisionTree', DecisionTreeClassifier(random_state=256)), 
    ('MLP', MLPClassifier(hidden_layer_sizes=(50, 50, 50, 50), activation='relu', learning_rate='adaptive', random_state=256)),
    ('KNClassifier', KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1))   
]


# Pipeline without resampling
estimator = LogisticRegression()
n_features = 15
direction = 'forward'
cv = 5

pipe_no_resampling = Pipeline(steps=[
    ('feature_selector', SequentialFeatureSelector(estimator=estimator, n_features_to_select=n_features, 
                                direction=direction, cv=5, n_jobs=-1)), 
    ('scaler', StandardScaler())])


# Pipeline with resampling
pipe_resampling = Pipeline(steps=[
    ('feature_selector', SequentialFeatureSelector(estimator=estimator, n_features_to_select=n_features, 
                                direction=direction, cv=5, n_jobs=-1)),
    ('smote', SMOTE(sampling_strategy=0.333, random_state=256, k_neighbors=3, n_jobs=-1)),
    ('scaler', StandardScaler())])


# Iterating over the classifiers and fitting

# dataset with 3d month only 
X_train = train_3_clean.loc[:, train_3_clean.columns != 'target_month3']
y_train = train_3_clean.loc[:, 'target_month3']

# Cross-validation parameters
scoring = ['average_precision', 'f1', 'accuracy', 'roc_auc', 'recall']
cv = StratifiedKFold(n_splits=5)

# lists where fitted classifiers with their metrics will be stored
fitted_classifiers_no_resampling = []
fitted_classifiers_resampling = []
for k in classifiers:
    temp_no_res = deepcopy(pipe_no_resampling)
    temp_no_res.steps.append(k)

    temp_res = deepcopy(pipe_resampling)
    temp_res.steps.append(k)

    fitted_classifiers_resampling.append(cross_validate(temp_res, X_train, y_train, 
                scoring=scoring, cv=cv, n_jobs=-1, return_estimator=True))
    fitted_classifiers_no_resampling.append(cross_validate(temp_no_res, X_train, y_train, 
                scoring=scoring, cv=cv, n_jobs=-1, return_estimator=True))


# Performance metrics for classifiers with resampled and non-resampled data

# non-resampled dataset
avg_precision_no_res = {}
f1_score_no_res = {}
accuracy_no_res = {}
roc_auc_no_res = {}
recall_no_res = {}
for i, cl in enumerate(fitted_classifiers_resampling):
    cl_performance = itemgetter('test_average_precision', 'test_f1', 'test_accuracy', 'test_roc_auc', 'test_recall')(cl)
    str_performance = f'''{classifiers[i][0]} performance: 
        test_average_precision: {max(cl_performance[0])}
        test_f1: {max(cl_performance[1])}
        test_accuracy: {max(cl_performance[2])}
        test_roc_auc: {max(cl_performance[3])}
        test_recall: {max(cl_performance[4])}'''

    avg_precision_no_res.update({classifiers[i][0] : max(cl_performance[0])})
    f1_score_no_res.update({classifiers[i][0] : max(cl_performance[1])})
    accuracy_no_res.update({classifiers[i][0] : max(cl_performance[2])})
    roc_auc_no_res.update({classifiers[i][0] : max(cl_performance[3])})
    recall_no_res.update({classifiers[i][0] : max(cl_performance[4])})

    # print(str_performance)

# resampled dataset
avg_precision_res = {}
f1_score_res = {}
accuracy_res = {}
roc_auc_res = {}
recall_res = {}
for i, cl in enumerate(fitted_classifiers_no_resampling):
    cl_performance = itemgetter('test_average_precision', 'test_f1', 'test_accuracy', 'test_roc_auc', 'test_recall')(cl)
    str_performance = f'''{classifiers[i][0]} performance: 
        test_average_precision: {max(cl_performance[0])}
        test_f1: {max(cl_performance[1])}
        test_accuracy: {max(cl_performance[2])}
        test_roc_auc: {max(cl_performance[3])}
        test_recall: {max(cl_performance[4])}'''

    avg_precision_res.update({classifiers[i][0] : max(cl_performance[0])})
    f1_score_res.update({classifiers[i][0] : max(cl_performance[1])})
    accuracy_res.update({classifiers[i][0] : max(cl_performance[2])})
    roc_auc_res.update({classifiers[i][0] : max(cl_performance[3])})
    recall_res.update({classifiers[i][0] : max(cl_performance[4])})

    # print(str_performance)

# Best classifier performances for each metric

perf_str = f'''
For resampled dataset:
Best average precision: {max(avg_precision_res, key=avg_precision_res.get)} = {avg_precision_res[max(avg_precision_res, key=avg_precision_res.get)]}
Best f1 score: {max(f1_score_res, key=f1_score_res.get)} = {f1_score_res[max(f1_score_res, key=f1_score_res.get)]}
Best accuracy: {max(accuracy_res, key=accuracy_res.get)} = {accuracy_res[max(accuracy_res, key=accuracy_res.get)]}
Best roc_auc: {max(roc_auc_res, key=roc_auc_res.get)} = {roc_auc_res[max(roc_auc_res, key=roc_auc_res.get)]}
Best recall: {max(recall_res, key=recall_res.get)} = {recall_res[max(recall_res, key=recall_res.get)]}
'''

print(perf_str)

perf_str = f'''
For original dataset
Best average precision: {max(avg_precision_no_res, key=avg_precision_no_res.get)} = {avg_precision_no_res[max(avg_precision_no_res, key=avg_precision_no_res.get)]}
Best f1 score: {max(f1_score_no_res, key=f1_score_no_res.get)} = {f1_score_no_res[max(f1_score_no_res, key=f1_score_no_res.get)]}
Best accuracy: {max(accuracy_no_res, key=accuracy_no_res.get)} = {accuracy_no_res[max(accuracy_no_res, key=accuracy_no_res.get)]}
Best roc_auc: {max(roc_auc_no_res, key=roc_auc_no_res.get)} = {roc_auc_no_res[max(roc_auc_no_res, key=roc_auc_no_res.get)]}
Best recall: {max(recall_no_res, key=recall_no_res.get)} = {recall_no_res[max(recall_no_res, key=recall_no_res.get)]}
'''

print(perf_str)

fitted_classifiers_resampling[2]

# make predictions for the test dataset with the best estimator

# Get based classifier from list 
# first [] : index of list - choose classifier
# third [] : index of list - choose kfold classifier with best performance
test_estimator = fitted_classifiers_resampling[2]['estimator'][0]

# columns that the SequentialFeatureSelector chose
cols_to_keep = test_estimator['feature_selector'].get_feature_names_out()

# Choose which values to predict
pred = test_estimator.predict(test_3_clean)

# Construction of .csv containing predictions
pred_df = pd.DataFrame(pred, columns=['predictions'])
pred_df['client_id'] = test_3_clean.index
pred_df.set_index('client_id', inplace=True)
pred_df.to_csv('data\\predictions.csv')