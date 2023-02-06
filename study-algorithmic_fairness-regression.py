#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 01:33:20 2022

@author: yoonsangcho

#regression: https://medium.com/responsibleml/what-fairness-in-regression-285e3f2a549e
#regression: https://dalex.drwhy.ai/python-dalex-fairness-regression.html
#classification: https://medium.com/responsibleml/how-to-easily-check-if-your-ml-model-is-fair-2c173419ae4c
#classification: https://dalex.drwhy.ai/python-dalex-fairness.html

"""

#%%
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# importing dalex to explain models
import dalex as dx 

data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data", header=None, na_values=["?"])
from urllib.request import urlopen
names = urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names")
columns = [line.split(b' ')[1].decode("utf-8") for line in names if line.startswith(b'@attribute')]
data.columns = columns
data = data.dropna(axis = 1)
data = data.iloc[:, 3:]

X = data.drop('ViolentCrimesPerPop', axis=1)
y = data.ViolentCrimesPerPop

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#%%
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

model2 = DecisionTreeRegressor()
model2.fit(X_train, y_train)

exp = dx.Explainer(model, X_test, y_test, verbose=False)
exp2 = dx.Explainer(model2, X_test, y_test, verbose=False)

print(exp.model_performance().result.append(exp2.model_performance().result))

'''
'''
protected = np.where(X_test.racepctblack >= 0.5, 'majority_black', "else")
privileged = 'else'

fobject = exp.model_fairness(protected, privileged)
fobject2 = exp2.model_fairness(protected, privileged)

fobject.fairness_check()
fobject2.fairness_check()

fobject2.plot()
fobject2.plot(fobject)

fobject.plot(fobject2, type='density')



