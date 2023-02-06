#%% cramersv
import numpy as np
import scipy.stats as stats
'''
https://www.geeksforgeeks.org/how-to-calculate-cramers-v-in-python/

Cramer’s V: It is defined as the measurement of length between two given nominal variables. 
A nominal variable is a type of data measurement scale that is used to categorize the different types of data. 
Cramer’s V lies between 0 and 1 (inclusive). 
0 indicates that the two variables are not linked by any relation. 
1 indicates that there exists a strong association between the two variables. 
Cramer’s V can be calculated by using the below formula:
'''

def cmr_v(dataset):
    # Finding Chi-squared test statistic, sample size, and minimum of rows  and columns
    X2 = stats.chi2_contingency(dataset, correction=False)[0]
    N = np.sum(dataset)
    min_dim = min(dataset.shape)-1
    cramer_v = np.sqrt((X2/N) / min_dim) # Calculate Cramer's V
    print('chi:', X2)
    print('N:', N, 'min_dim:', min_dim)
    print("Cramer's V: ", cramer_v)
    return cramer_v

def stat_chi(df_cross):
    # df_cross = df_adult_cross.copy()
    I = range(df_cross.shape[0])
    J = range(df_cross.shape[1])
    N = np.sum(df_cross)
    chi = 0
    for i in I:
        for j in J:
            nij = df_cross[i, j]
            ni_ = df_cross[i, :].sum()
            n_j = df_cross[:, j].sum()
            chi += ((nij-((ni_*n_j)/N))**2) / ((ni_*n_j)/N)
    # print('chi:', chi)
    return chi

def cmr_v_(df_cross):
    N = np.sum(df_cross)
    chi = stat_chi(df_cross)
    min_dim = min(df_cross.shape)-1
    cramer_v = np.sqrt((chi/N) / min_dim).round(2) # Calculate Cramer's V
    print('chi:', chi)
    print('N:', N, 'min_dim:', min_dim)
    print("Cramer's V: ", cramer_v)
    return cramer_v

dataset = np.array([[100, 100, 100], 
                    [100, 100, 100]
                    ])
cmr_v(dataset)
cmr_v_(dataset)

dataset = np.array([[0, 1, 100], 
                    [100, 1, 0]
                    ])
cmr_v(dataset)
cmr_v_(dataset)

dataset = np.array([[0, 100], 
                    [100, 0]])
cmr_v(dataset)
cmr_v_(dataset)

dataset = np.array([[100, 0], 
                    [0, 100]])
cmr_v(dataset)
cmr_v_(dataset)

dataset = np.array([[100, 100], 
                    [50, 100]])
cmr_v(dataset)
cmr_v_(dataset)

dataset = np.array([[100, 50], 
                    [50, 100]])
cmr_v(dataset)
cmr_v_(dataset)


dataset = np.array([[50, 100], 
                    [100, 50]])
cmr_v(dataset)
cmr_v_(dataset)

dataset = np.array([[100, 0], 
                    [50, 100]])
cmr_v(dataset)
cmr_v_(dataset)

dataset = np.array([[0, 20], 
                    [10, 10]])

cmr_v(dataset)
cmr_v_(dataset)

dataset = np.array([[100, 50], 
                    [0, 100]])

cmr_v(dataset)
cmr_v_(dataset)

dataset = np.array([[50, 100], 
                    [100, 0]])

cmr_v(dataset)
cmr_v_(dataset)

dataset = np.array([[100, 100], 
                    [100, 100]])
cmr_v(dataset)
cmr_v_(dataset)

#%%
# Make a 3 x 3 table
dataset = np.array([[13, 17, 11], 
                    [4, 6, 9],
                    [20, 31, 42]])

# Finding Chi-squared test statistic, sample size, and minimum of rows  and columns
X2 = stats.chi2_contingency(dataset, correction=False)[0]
N = np.sum(dataset)
minimum_dimension = min(dataset.shape)-1
result = np.sqrt((X2/N) / minimum_dimension) # Calculate Cramer's V
print(result) # Print the result

# Example 2:
'''
We will now calculate Cramer’s V for larger tables and having unequal dimensions. 
The Cramers V comes out to be equal to 0.12 which clearly depicts the weak association between the two variables in the table.
'''
import scipy.stats as stats
import numpy as np
  
# Make a 5 x 4 table
dataset = np.array([[4, 13, 17, 11], 
                    [4, 6, 9, 12],
                    [2, 7, 4, 2], 
                    [5, 13, 10, 12],
                    [5, 6, 14, 12]])
  
# Finding Chi-squared test statistic, 
# sample size, and minimum of rows and
# columns
X2 = stats.chi2_contingency(dataset, correction=False)[0]
N = np.sum(dataset)
minimum_dimension = min(dataset.shape)-1
  
# Calculate Cramer's V
result = np.sqrt((X2/N) / minimum_dimension)
  
# Print the result
print(result)