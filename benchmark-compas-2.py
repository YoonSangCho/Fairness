#%% LSAC DATSET
import os
import tempfile
import pandas as pd
import six.moves.urllib as urllib
import pprint
# import tensorflow_model_analysis as tfma
from google.protobuf import text_format
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
# Download the LSAT dataset and setup the required filepaths.
_DATA_ROOT = tempfile.mkdtemp(prefix='lsat-data')
_DATA_PATH = 'https://storage.googleapis.com/lawschool_dataset/bar_pass_prediction.csv'
_DATA_FILEPATH = os.path.join(_DATA_ROOT, 'bar_pass_prediction.csv')

data = urllib.request.urlopen(_DATA_PATH)

_LSAT_DF = pd.read_csv(data)
_LSAT_DF.to_csv('data/lsat_raw.csv')

_LSAT_DF.columns

# To simpliy the case study, we will only use the columns that will be used for our model.
_COLUMN_NAMES = [
  'dnn_bar_pass_prediction',
  'gender',
  'lsat',
  'pass_bar',
  'race1',
  'ugpa',
]

_LSAT_DF.dropna()
_LSAT_DF['gender'] = _LSAT_DF['gender'].astype(str)
_LSAT_DF['race1'] = _LSAT_DF['race1'].astype(str)
_LSAT_DF = _LSAT_DF[_COLUMN_NAMES]

_LSAT_DF.head()


#%% COMPAS Dataset
# ref: http://blog.kennylee.info/projects/python/data/machinelearning/bias/2020/11/01/analyze-Compas.html
data_URL = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
data_raw = pd.read_csv(data_URL)
# data_raw.to_csv('data/compas-scores-two-years.csv.csv')
# Select fields for severity of charge, number of priors, demographics, age, sex, compas scores, and whether each person was accused of a crime within two years.
colnames_all = list(data_raw.columns)
colnames = ['sex', 'age_cat', 'race', #'age', 
            'juv_fel_count','juv_misd_count','juv_other_count',
            'priors_count', 'priors_count.1',
            'c_charge_degree', 
            'two_year_recid', #'is_recid', 'is_violent_recid', 
            #'c_days_from_compas', 'r_days_from_arrest', 'days_b_screening_arrest',
            'decile_score', 'decile_score.1','v_decile_score',
            'score_text','v_score_text',            
            'start','end','event',
            ]

assert data_raw[colnames].isna().sum().sum() == 0
data = data_raw[colnames].copy()
data.info()

colnames_s = ['sex', 'race', 'age_cat']
colnames_x_num = ['juv_fel_count','juv_misd_count','juv_other_count',
            'priors_count', 
            'decile_score', 'v_decile_score',
            'start','end','event',
            ]
colnames_x_fac = ['score_text','v_score_text','c_charge_degree']
colnames_y = ['two_year_recid']
data[colnames_s] = data[colnames_s].astype('category')
data[colnames_x_fac] = data[colnames_x_fac].astype('category')

s = data[colnames_s]
sexes = np.unique(s['sex'])
races = np.unique(s['race'])
agees = np.unique(s['age_cat'])

s['sex'].value_counts().plot(kind='barh', )
plt.show()

s['race'].value_counts().plot(kind='barh', )
plt.show()

s['age_cat'].value_counts().plot(kind='barh', )
plt.show()

x = data[colnames_x_num+colnames_x_fac]
y = data[colnames_y]

x_dum = pd.get_dummies(data = x, columns=colnames_x_fac, drop_first=False)
s_dum = pd.get_dummies(data = s, columns=colnames_s, drop_first=False)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train, x_test, s_train, s_test, y_train, y_test = train_test_split(x_dum, s_dum, y, test_size=0.2, random_state=0, stratify=s_dum['sex_Male'])


sum(s_train['sex_Female'])
sum(s_test['sex_Female'])
sum(s_test['sex_Male'])

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model_y = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr',random_state=0)
model_y.fit(x_train, y_train)

y_pred = model_y.predict(x_test)
print('model_y.score(x_train, y_train)', model_y.score(x_train, y_train))
print('model_y.score(x_test, y_test)', model_y.score(x_test, y_test))

model_s = LogisticRegression(random_state=0)
model_s.fit(x_train, s_train['sex_Male'])

print("model_s.score(x_train, s_train['sex_Male'])", model_s.score(x_train, s_train['sex_Male']))
print("model_s.score(x_test, s_test['sex_Male'])", model_s.score(x_test, s_test['sex_Male']))

s_pred_train = model_s.predict(x_train)
s_pred_train_prob = model_s.predict_proba(x_train)
s_pred_train_logp = model_s.predict_log_proba(x_train)

s_test
s_pred_test = model_s.predict(x_test)
s_pred_test_prob = model_s.predict_proba(x_test)
s_pred_test_logp = model_s.predict_log_proba(x_test)
pd.DataFrame(s_pred_ce)


def cross_entropy_loss(y_pred, y):
    if y == 1:
      return -np.log(y_pred)
    else:
      return -np.log(1 - y_pred)

s_pred_ce = []  
for i in range(len(s_test)):
    ce = cross_entropy_loss(s_pred_test_prob[i, 1], np.array(s_test['sex_Male'])[i])
    s_pred_ce.append(ce)
-np.log(1 - s_pred_test_prob[i, 1])
sum(s_pred_test==0)
model_s.score(x_train[0], s_train['sex_Male'][:1])

import numpy as np
import matplotlib.pyplot as plt
 
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
 
'''
yHat represents the predicted value / probability value calculated as output of hypothesis / sigmoid function 
y represents the actual label
'''
def cross_entropy_loss(y_pred, y):
    if y == 1:
      return -np.log(y_pred)
    else:
      return -np.log(1 - y_pred)
  
# colnames_example = ['sex', 'race', 'age', 'age_cat', 'score_text', 'priors_count', 'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']
# colnames_x_example = ['score_text_medhi ~ sex_female + age_cat_greater_than_45 + age_cat_less_than_25 + race_african_american + race_asian + race_hispanic + race_native_american + race_other + priors_count + c_charge_degree_m + two_year_recid']
# conlnames = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']

data_raw = data_raw[conlnames]

dfFiltered = (data_raw[conlnames]
             .loc[(data_raw['days_b_screening_arrest'] <= 30) & (dfRaw['days_b_screening_arrest'] >= -30), :]
             .loc[data_raw['is_recid'] != -1, :]
             .loc[data_raw['c_charge_degree'] != 'O', :]
             .loc[data_raw['score_text'] != 'N/A', :]
             )
dfFiltered = (data_raw[conlnames]
             .loc[(data_raw['days_b_screening_arrest'] <= 30) & (dfRaw['days_b_screening_arrest'] >= -30), :]
             .loc[data_raw['is_recid'] != -1, :]
             .loc[data_raw['c_charge_degree'] != 'O', :]
             .loc[data_raw['score_text'] != 'N/A', :]
             )

print('Number of rows: {}'.format(len(dfFiltered.index)))
pd.crosstab(dfFiltered['score_text'],dfFiltered['race'])
pd.crosstab(dfFiltered['score_text'],dfFiltered['decile_score'])

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='darkgrid')
sns.countplot(x='decile_score', hue='race', data=dfFiltered.loc[
                (dfFiltered['race'] == 'African-American') | (dfFiltered['race'] == 'Caucasian'),:
            ])

plt.title("Distribution of Decile Scores by Race")
plt.xlabel('Decile Score')
plt.ylabel('Count')
plt.show()


import statsmodels.api as sm
from statsmodels.formula.api import logit
catCols = ['score_text','age_cat','sex','race','c_charge_degree']
dfFiltered.loc[:,catCols] = dfFiltered.loc[:,catCols].astype('category')

# dfDummies = pd.get_dummies(data = dfFiltered.loc[dfFiltered['score_text'] != 'Low',:], columns=catCols)
dfDummies = pd.get_dummies(data = dfFiltered, columns=catCols)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Clean column names
new_column_names = [col.lstrip().rstrip().lower().replace(" ", "_").replace("-", "_") for col in dfDummies.columns]
dfDummies.columns = new_column_names

# We want another variable that combines Medium and High
dfDummies['score_text_medhi'] = dfDummies['score_text_medium'] + dfDummies['score_text_high']
np.unique(dfDummies['score_text_medhi'])
# Logistic regression

formula = 'score_text_medhi ~ sex_female + age_cat_greater_than_45 + age_cat_less_than_25 + race_african_american + race_asian + race_hispanic + race_native_american + race_other + priors_count + c_charge_degree_m + two_year_recid'
dfDummies['two_year_recid']

score_mod = logit(formula, data = dfDummies).fit()
print(score_mod.summary())



#%% Binary Classification on COMPAS dataset
# https://fairlearn.org/v0.4.6/auto_examples/plot_binary_classification_COMPAS.html

import pandas as pd
import numpy as np
from tempeh.configurations import datasets

compas_dataset = datasets["compas"]()
X_train, X_test = compas_dataset.get_X(format=pd.DataFrame)
y_train, y_test = compas_dataset.get_y(format=pd.Series)
(
    sensitive_features_train,
    sensitive_features_test,
) = compas_dataset.get_sensitive_features("race", format=pd.Series)
X_train.loc[0], y_train[0]

(train_s_sex, test_s_sex) = compas_dataset.get_sensitive_features("sex", format=pd.Series)

compas_dataset = datasets["compas"]()
compas_dataset.__dir__()
X_train, X_test = compas_dataset.get_X(format=pd.DataFrame)
y_train, y_test = compas_dataset.get_y(format=pd.Series)
(sensitive_features_train, sensitive_features_test,) = compas_dataset.get_sensitive_features("race", format=pd.Series)

(train_s_race, test_s_race,) = compas_dataset.get_sensitive_features("race", format=pd.Series)
(train_s_sex, test_s_sex) = compas_dataset.get_sensitive_features("sex", format=pd.Series)


X_train.loc[0], y_train[0]

#%%
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# show_proportions is only a helper function for plotting
def show_proportions(
    X, sensitive_features, y_pred, y=None, description=None, plot_row_index=1
):
    print("\n" + description)
    plt.figure(plot_row_index)
    plt.title(description)
    plt.ylabel("P[recidivism predicted | conditions]")

    indices = {}
    positive_indices = {}
    negative_indices = {}
    recidivism_count = {}
    recidivism_pct = {}
    groups = np.unique(sensitive_features.values)
    n_groups = len(groups)
    max_group_length = max([len(group) for group in groups])
    color = cm.rainbow(np.linspace(0, 1, n_groups))
    x_tick_labels_basic = []
    x_tick_labels_by_label = []
    for index, group in enumerate(groups):
        indices[group] = sensitive_features.index[sensitive_features == group]
        recidivism_count[group] = sum(y_pred[indices[group]])
        recidivism_pct[group] = recidivism_count[group] / len(indices[group])
        print(
            "P[recidivism predicted | {}]                {}= {}".format(
                group, " " * (max_group_length - len(group)), recidivism_pct[group]
            )
        )

        plt.bar(index + 1, recidivism_pct[group], color=color[index])
        x_tick_labels_basic.append(group)

        if y is not None:
            positive_indices[group] = sensitive_features.index[
                (sensitive_features == group) & (y == 1)
            ]
            negative_indices[group] = sensitive_features.index[
                (sensitive_features == group) & (y == 0)
            ]
            prob_1 = sum(y_pred[positive_indices[group]]) / len(positive_indices[group])
            prob_0 = sum(y_pred[negative_indices[group]]) / len(negative_indices[group])
            print(
                "P[recidivism predicted | {}, recidivism]    {}= {}".format(
                    group, " " * (max_group_length - len(group)), prob_1
                )
            )
            print(
                "P[recidivism predicted | {}, no recidivism] {}= {}".format(
                    group, " " * (max_group_length - len(group)), prob_0
                )
            )

            plt.bar(n_groups + 1 + 2 * index, prob_1, color=color[index])
            plt.bar(n_groups + 2 + 2 * index, prob_0, color=color[index])
            x_tick_labels_by_label.extend(
                ["{} recidivism".format(group), "{} no recidivism".format(group)]
            )

    x_tick_labels = x_tick_labels_basic + x_tick_labels_by_label
    plt.xticks(
        range(1, len(x_tick_labels) + 1),
        x_tick_labels,
        rotation=20,
        horizontalalignment="right",
    )
    plt.show()
    
#%% LogisticRegression
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression(solver="liblinear") # solver{‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’}, default=’lbfgs’
estimator.fit(X_train, y_train)

show_proportions(X_train, 
                 sensitive_features_train,
                 y_train,
                 description="original training data:", plot_row_index=1)

show_proportions(X_train, 
                 sensitive_features_train,
                 estimator.predict(X_train), 
                 y_train,
                 description="fairness-unaware prediction on training data:",
                 plot_row_index=2)


show_proportions(X_test,
                 sensitive_features_test,
                 y_test,
                 description="original test data:",
                 plot_row_index=3)

show_proportions(X_test,
                 sensitive_features_test,
                 estimator.predict(X_test),
                 y_test,
                 description="fairness-unaware prediction on test data:",
                 plot_row_index=4,)


plt.show()

