#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 01:06:56 2022

@author: yoonsangcho

ref: https://towardsdatascience.com/compas-case-study-fairness-of-a-machine-learning-model-f0f804108751
"""

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

url = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
df = pd.read_csv(url)
df.info()

# turn into a binary classification problem
# create feature is_med_or_high_risk
df['is_med_or_high_risk']  = (df['decile_score']>=5).astype(int)

# classification accuracy
np.mean(df['is_med_or_high_risk']==df['two_year_recid'])
#0.6537288605489326
np.mean(df['two_year_recid'])
#0.45065151095092876


cm = pd.crosstab(df['is_med_or_high_risk'], df['two_year_recid'], rownames=['Predicted'], colnames=['Actual'])
p = plt.figure(figsize=(5,5));
p = sns.heatmap(cm, annot=True, fmt="d", cbar=False)
plt.show()

[[tn , fp],[fn , tp]]  = confusion_matrix(df['two_year_recid'], df['is_med_or_high_risk'])
print("True negatives:  ", tn)
print("False positives: ", fp)
print("False negatives: ", fn)
print("True positives:  ", tp)

cm = pd.crosstab(df['is_med_or_high_risk'], df['two_year_recid'], rownames=['Predicted'], colnames=['Actual'], normalize='index')
p = plt.figure(figsize=(5,5));
p = sns.heatmap(cm, annot=True, fmt=".2f", cbar=False)

cm = pd.crosstab(df['is_med_or_high_risk'], df['two_year_recid'], rownames=['Predicted'], colnames=['Actual'], normalize='columns')
p = plt.figure(figsize=(5,5));
p = sns.heatmap(cm, annot=True, fmt=".2f", cbar=False)

# compute?
fpr = fp/(fp+tn)
fnr  = fn/(fn+tp)

print("False positive rate (overall): ", fpr)
print("False negative rate (overall): ", fnr)

d = df.groupby('decile_score').agg({'two_year_recid': 'mean'})
# plot
sns.scatterplot(data=d);
plt.ylim(0,1);
plt.ylabel('Recidivism rate');


from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(df['two_year_recid'], df['decile_score'])
sns.scatterplot(x=fpr, y=tpr, );
sns.lineplot(x=fpr, y=tpr);
plt.ylabel("TPR");
plt.xlabel("FPR");


auc = roc_auc_score(df['two_year_recid'], df['decile_score'])
auc

#%%
#Fairness: COMPAS has been under scrutiny for issues related for fairness with respect to race of the defendant. 
# Race is not an explicit input to COMPAS, but some of the questions that are used as input may have strong correlations with race.
# First, we will find out how frequently each race is represented in the data:
df['race'].value_counts()













