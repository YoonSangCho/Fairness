#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 23:53:56 2022

@author: yoonsangcho
ref: https://techairesearch.com/most-essential-python-fairness-libraries-every-data-scientist-should-know/

ref: https://github.com/fairlearn/fairlearn
ref: https://fairlearn.org/v0.7.0/auto_examples/index.html
"""

#%% Selection rates in census dataset
from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.datasets import fetch_adult

data = fetch_adult(as_frame=True)
X = data.data
y_true = (data.target == '>50K') * 1
sex = X['sex']

selection_rates = MetricFrame(metrics=selection_rate,
                              y_true=y_true,
                              y_pred=y_true,
                              sensitive_features=sex)

fig = selection_rates.by_group.plot.bar(
    legend=False, rot=0,
    title='Fraction earning over $50,000')



#%% MetricFrame visualizations¶
from fairlearn.metrics import (
    MetricFrame,
    false_positive_rate,
    true_positive_rate,
    selection_rate,
    count
)
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

data = fetch_openml(data_id=1590, as_frame=True)
colnames = data.data.columns
X = pd.get_dummies(data.data)
y_true = (data.target == ">50K") * 1
sex = data.data["sex"]

classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
classifier.fit(X, y_true)
y_pred = classifier.predict(X)

# Analyze metrics using MetricFrame
metrics = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'false positive rate': false_positive_rate,
    'true positive rate': true_positive_rate,
    'selection rate': selection_rate,
    'count': count}
metric_frame = MetricFrame(metrics=metrics,
                           y_true=y_true,
                           y_pred=y_pred,
                           sensitive_features=sex)
metric_frame.by_group.plot.bar(
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 8],
    title="Show all metrics",
)

# Customize plots with ylim
metric_frame.by_group.plot(
    kind="bar",
    ylim=[0, 1],
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 8],
    title="Show all metrics with assigned y-axis range",
)

# Customize plots with colormap
metric_frame.by_group.plot(
    kind="bar",
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 8],
    colormap="Accent",
    title="Show all metrics in Accent colormap",
)

# Customize plots with kind
metric_frame.by_group.plot(
    kind="pie",
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 8],
    title="Show all metrics in pie",
)

#%% Making Derived Metrics¶
import functools

import numpy as np
import sklearn.metrics as skm
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from fairlearn.metrics import MetricFrame, make_derived_metric
from fairlearn.metrics import accuracy_score_group_min

# UCI ‘Adult’ dataset: Next, we import the data, dropping any rows which are missing data:
data = fetch_openml(data_id=1590, as_frame=True)
X_raw = data.data
colnames = X_raw.columns
y = (data.target == ">50K") * 1
# np.unique(y)
A = X_raw[["race", "sex"]]

(X_train, X_test, y_train, y_test, A_train, A_test) = train_test_split(
    X_raw, y, A, test_size=0.3, random_state=12345, stratify=y
)

# Ensure indices are aligned between X, y and A,
# after all the slicing and splitting of DataFrames
# and Series

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
A_train = A_train.reset_index(drop=True)
A_test = A_test.reset_index(drop=True)

numeric_transformer = Pipeline(
    steps=[
        ("impute", SimpleImputer()),
        ("scaler", StandardScaler()),
    ]
)
categorical_transformer = Pipeline(
    [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, selector(dtype_exclude="category")),
        ("cat", categorical_transformer, selector(dtype_include="category")),
    ]
)

# Logistic Regression
unmitigated_predictor = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(solver="liblinear", fit_intercept=True),
        ),
    ]
)
unmitigated_predictor.fit(X_train, y_train)
y_pred = unmitigated_predictor.predict(X_test)

# Creating a derived metric

acc_frame = MetricFrame(
    metrics=skm.accuracy_score,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test["sex"]
)
print("Minimum accuracy_score: ", acc_frame.group_min())

my_acc = make_derived_metric(metric=skm.accuracy_score, transform="group_min")
my_acc_min = my_acc(y_test, y_pred, sensitive_features=A_test["sex"])
print("Minimum accuracy_score: ", my_acc_min)

##### random weights
random_weights = np.random.rand(len(y_test))

acc_frame_sw = MetricFrame(
    metrics=skm.accuracy_score,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test["sex"],
    sample_params={"sample_weight": random_weights},
)

from_frame = acc_frame_sw.group_min()
from_func = my_acc(
    y_test,
    y_pred,
    sensitive_features=A_test["sex"],
    sample_weight=random_weights,
)

print("From MetricFrame:", from_frame)
print("From function   :", from_func)



##### fbeta
fbeta_03 = functools.partial(skm.fbeta_score, beta=0.3)
fbeta_03.__name__ = "fbeta_score__beta_0.3"

beta_frame = MetricFrame(
    metrics=fbeta_03,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test["sex"],
    sample_params={"sample_weight": random_weights},
)
beta_from_frame = beta_frame.difference(method="to_overall")

print("From frame:", beta_from_frame)

beta_func = make_derived_metric(metric=skm.fbeta_score, transform="difference")

beta_from_func = beta_func(
    y_test,
    y_pred,
    sensitive_features=A_test["sex"],
    beta=0.3,
    sample_weight=random_weights,
    method="to_overall",
)

print("From function:", beta_from_func)



##### Pregenerated Metrics
from_myacc = my_acc(y_test, y_pred, sensitive_features=A_test["race"])

from_pregen = accuracy_score_group_min(
    y_test, y_pred, sensitive_features=A_test["race"]
)

print("From my function :", from_myacc)
print("From pregenerated:", from_pregen)
assert from_myacc == from_pregen

#%%






















