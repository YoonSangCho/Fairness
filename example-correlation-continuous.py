import numpy as np
import pandas as pd

'''
ref: https://towardsdatascience.com/independence-covariance-and-correlation-between-two-random-variables-197022116f93
'''
#%% Independence
def create_marginal_pmfs(distr_table: pd.DataFrame):
    range_k = distr_table['K'].unique()
    range_j = distr_table['J'].unique()
    pk_marginal = dict()
    pj_marginal = dict()
    for k in range_k:
        # k=range_k[0]
        temp = 0
        for j in range_j:
            # j=range_j[0]
            temp += distr_table[(distr_table.K==k) & (distr_table.J==j)].pr.iloc[0]
        pk_marginal[k] = temp

    for j in range_j:
        temp = 0
        for k in range_k:
            temp += distr_table[(distr_table.K==k) & (distr_table.J==j)].pr.iloc[0]
        pj_marginal[j] = temp
        
    return pk_marginal, pj_marginal

def check_independence(distr_table: pd.DataFrame):
    pk_marginal, pj_marginal = create_marginal_pmfs(distr_table)
    independent = True
    # k=range_k[0]
    # j=range_j[0]
    for k in pk_marginal:
        for j in pj_marginal:            
            ##### if for **ANY** x, y : p(x,y) != p(x)*p(y) the random variables are not independent
            if (pk_marginal[k]*pj_marginal[j] != distr_table[(distr_table.K==k) & (distr_table.J==j)].pr.iloc[0]):
            # if (pk_marginal[k]*pj_marginal[j] != round(distr_table[(distr_table.K==k) & (distr_table.J==j)].pr.iloc[0],3)):
                independent = False
            # print('independent', independent)
    return independent


#%% covariance (between two variables: continuous)
def calculate_covariance(distr_table):
    pk_marginal, pj_marginal = create_marginal_pmfs(distr_table)
    mu_k = sum([k*p for k,p in pk_marginal.items()])
    mu_j = sum([j*p for j,p in pj_marginal.items()])

    cov_kj = 0

    for k in pk_marginal:
        for j in pj_marginal:
            cov_kj += (k-mu_k)*(j-mu_j)*distr_table[(distr_table.K==k) & (distr_table.J==j)].pr.iloc[0]
    return cov_kj
#%% correlation (between two variables: continuous)
def calculate_correlation(distr_table: pd.DataFrame):
    pk_marginal, pj_marginal = create_marginal_pmfs(distr_table)
    cov_kj = calculate_covariance(distr_table)
    mu_k = sum([k*p for k,p in pk_marginal.items()])/len(pk_marginal)
    mu_j = sum([j*p for j,p in pj_marginal.items()])/len(pj_marginal)

    var_k = 0
    var_j = 0

    for k, pk in pk_marginal.items():
        var_k += pk*(k-mu_k)**2

    for j, pj in pj_marginal.items():
        var_j += pj*(j-mu_j)**2
    corr_kj = cov_kj / (var_k*var_j)
    return corr_kj

#%% Independence, Covariance, Correlation
# not independent
distr_table_1 = pd.DataFrame({
    'K': [0, 0, 0, 0, 1, 1, 1, 1],
    'J': [0, 1, 2, 3, 0, 1, 2, 3],
    'pr': [1/8, 2/8, 1/8, 0, 0, 1/8, 2/8, 1/8]
})
print("distr_table_1:", distr_table_1)
check_independence(distr_table_1)
# distr_table = distr_table_1.copy() 

#independent
distr_table_2 = pd.DataFrame({
    'K': [0, 0, 0, 0, 1, 1, 1, 1],
    'J': [0, 1, 2, 3, 0, 1, 2, 3],
    'pr': [1/16, 3/16, 3/16, 1/16, 1/16, 3/16, 3/16, 1/16]
})
print("distr_table_2:", distr_table_2)
check_independence(distr_table_2)
# distr_table = distr_table_2.copy() 

distr_table_3 = pd.DataFrame({
    'K': [0, 0, 0, 0, 1, 1, 1, 1],
    'J': [0, 1, 2, 3, 0, 1, 2, 3],
    'pr': [2/16, 2/16, 2/16, 2/16, 2/16, 2/16, 2/16, 2/16]
})
print("distr_table_3:", distr_table_3)
check_independence(distr_table_3)
# distr_table = distr_table_3.copy() 


calculate_covariance(distr_table_1)
calculate_correlation(distr_table_1)


