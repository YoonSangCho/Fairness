#%% Modules and References
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import scipy.stats as stats

'''
ref: https://thinkingneuron.com/how-to-measure-the-correlation-between-two-categorical-variables-in-python/
'''


#%% ########## Benchmark Datasets
# (1) Adult
'''
### age: continuous.
### workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
### fnlwgt: continuous.
### education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
### education-num: continuous.
### marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
### occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
### relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
### race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
### sex: Female, Male.
### capital-gain: continuous.
### capital-loss: continuous.
### hours-per-week: continuous.
### native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
### class: >50K, <=50K
'''

df_adult = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data') 
colnames_adult = ['age', 'workclass','fnlwgt','education','education-num',
                  'marital-status','occupation','relationship','race',
                  'sex','capital-gain','capital-loss','hours-per-week',
                  'native-country','class']
df_adult.columns = colnames_adult
print(df_adult.sex.unique())
print(df_adult.race.unique())

# Contingency Tables: cross tabulation between GENDER and RACE
df_adult_cross=pd.crosstab(index=df_adult['sex'],columns=df_adult['race'])
print(df_adult_cross)

# (2) Benchmark: COMPAS
df_compas = pd.read_csv("data/compas_raw.csv")
df_compas.columns = df_compas.columns.str.replace(' ', '')
colnames_compas = list(df_compas.columns)
print(df_compas.sex.unique())
print(df_compas.race.unique())
# Contingency Tables: cross tabulation between GENDER and RACE
df_compas_cross=pd.crosstab(index=df_compas['sex'],columns=df_compas['race'])

# (3) Benchmark: LACS
df_lacs = pd.read_csv("data/LSAC_raw.csv").iloc[:,2:]
print(df_lacs.isna().sum())
df_lacs = df_lacs.dropna(axis='columns', thresh=900) # int(df_lacs.shape[0]*0.05)
df_lacs = df_lacs.dropna(axis='rows') # int(df_lacs.shape[0]*0.05)
print(df_lacs.isna().sum())
assert df_lacs.isna().sum().sum() == 0
colnames_lacs = list(df_lacs.columns)
# Contingency Tables: cross tabulation between GENDER and RACE
df_lacs_cross=pd.crosstab(index=df_lacs['sex'],columns=df_lacs['race'])


# (4) example: 2x2 groups
columns=['CIBIL','AGE','GENDER' ,'SALARY', 'APPROVE_LOAN']
values=[ [480, 28, 'M', 610000, 'Yes'],
             [480, 42, 'M',140000, 'No'],
             [480, 29, 'F',420000, 'No'],
             [490, 30, 'M',420000, 'No'],
             [500, 27, 'M',420000, 'No'],
             [510, 34, 'F',190000, 'No'],
             [550, 24, 'M',330000, 'Yes'],
             [560, 34, 'M',160000, 'Yes'],
             [560, 25, 'F',300000, 'Yes'],
             [570, 34, 'M',450000, 'Yes'],
             [590, 30, 'F',140000, 'Yes'],
             [600, 33, 'M',600000, 'Yes'],
             [600, 22, 'M',400000, 'Yes'],
             [600, 25, 'F',490000, 'Yes'],
             [610, 32, 'M',120000, 'Yes'],
             [630, 29, 'F',360000, 'Yes'],
             [630, 30, 'M',480000, 'Yes'],
             [660, 29, 'F',460000, 'Yes'],
             [700, 32, 'M',470000, 'Yes'],
             [740, 28, 'M',400000, 'Yes']]

columns = ['GENDER' ,'RACE']
values = [
    ['M', 'W'],['M', 'W'],['M', 'W'],['M', 'W'],['M', 'W'],
    ['M', 'W'],['M', 'W'],['M', 'W'],['M', 'W'],['M', 'W'],
    ['M', 'B'],['M', 'B'],['M', 'B'],['M', 'B'],['M', 'B'],
    ['M', 'B'],['M', 'B'],['M', 'B'],['M', 'B'],['M', 'B'],
    ['F', 'W'],['F', 'W'],['F', 'W'],['F', 'W'],['F', 'W'],
    ['F', 'W'],['F', 'W'],['F', 'W'],['F', 'W'],['F', 'W'],
    ['F', 'W'],['F', 'W'],['F', 'W'],['F', 'W'],['F', 'W'],
    ['F', 'W'],['F', 'W'],['F', 'W'],['F', 'W'],['F', 'W'],
          ]
# create the Data Frame
df_example = pd.DataFrame(data=values,columns=columns)
print(df_example.head())
# Contingency Tables: cross tabulation between GENDER and RACE
df_example_cross=pd.crosstab(index=df_example['GENDER'],columns=df_example['RACE'])
print(df_example_cross)

#%% ########## Correlation

"""
Chi-square test finds the probability of a Null hypothesis(H0).
Assumption(H0): The two columns are NOT related to each other
Result of Chi-Sq Test: The Probability of H0 being True
More information on ChiSq can be found here
"""

def cmr_v(df_cross):
    # df_cross = df_adult_cross.copy()
    res_chisq = chi2_contingency(df_cross, correction=False)
    # res_chisq = chi2_contingency(df_cross, correction=True)
    chi2 = res_chisq[0].round(2)
    p_val = res_chisq[1].round(2)
    N = df_cross.values.sum()
    min_dim = min(df_cross.shape)-1
    cramer_v = np.sqrt((chi2/N) / min_dim).round(2) # Calculate Cramer's V
    print('chi2:', res_chisq[0].round(2), 'p_val:', res_chisq[1].round(2))
    print('N:', N, 'min_dim:', min_dim)
    print("Cramer's V: ", cramer_v)
    return cramer_v

def stat_phi(df_cross):
    # df_cross = df_example_cross.copy()
    n00 = df_cross.iloc[0, 0]
    n01 = df_cross.iloc[0, 1]
    n10 = df_cross.iloc[1, 0]
    n11 = df_cross.iloc[1, 1]
    n1_ = n10+n11
    n_1 = n01+n11    
    n0_ = n00+n01
    n_0 = n00+n10
    phi = ((n11*n00)-(n01*n10))/np.sqrt(n0_*n_0*n1_*n_1)
    print('phi:', phi)
    return phi

def stat_chi(df_cross):
    # df_cross = df_adult_cross.copy()
    I = range(df_cross.shape[0])
    J = range(df_cross.shape[1])
    # N = np.sum(df_cross)
    N = df_cross.values.sum()
    chi = 0
    for i in I:
        for j in J:
            nij = df_cross.iloc[i, j]
            ni_ = df_cross.iloc[i, :].sum()
            n_j = df_cross.iloc[:, j].sum()
            chi += ((nij-((ni_*n_j)/N))**2) / ((ni_*n_j)/N)
    # print('chi:', chi)
    return chi

def cmr_v_(df_cross):
    chi = stat_chi(df_cross)
    # N = np.sum(df_cross)
    N = df_cross.values.sum()
    min_dim = min(df_cross.shape)-1
    cramer_v = np.sqrt((chi/N) / min_dim).round(2) # Calculate Cramer's V
    print('chi:', chi)
    print('N:', N, 'min_dim:', min_dim)
    print("Cramer's V: ", cramer_v)
    return cramer_v

corr_cmr_adult = cmr_v(df_adult_cross) # K(=2)xJ(>2)
corr_cmr_adult_ = cmr_v_(df_adult_cross)

corr_cmr_compas = cmr_v(df_compas_cross) # K(=2)xJ(>2)
corr_cmr_compas = cmr_v_(df_compas_cross) # K(=2)xJ(>2)

corr_cmr_lacs = cmr_v(df_lacs_cross) # K(=2)xJ(>2)
corr_cmr_lacs_ = cmr_v_(df_lacs_cross) # K(=2)xJ(>2)

corr_cmr_example  = cmr_v(df_example_cross) # K(=2)xJ(=2)
corr_cmr_example_ = cmr_v_(df_example_cross) # K(=2)xJ(=2)
