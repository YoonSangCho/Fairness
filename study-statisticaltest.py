#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

https://scikit-survival.readthedocs.io/en/stable/user_guide/understanding_predictions.html
https://scikit-survival.readthedocs.io/en/stable/user_guide/index.html
https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
http://bigdata.dongguk.ac.kr/lectures/med_stat/_book/%EC%83%9D%EC%A1%B4%EB%B6%84%EC%84%9D-survival-analysis.html

https://bioinformaticsandme.tistory.com/223


Created on Tue Mar 22 16:10:01 2022

@author: yoonsangcho

Statistical Hypothesis Tests: Tutorial Overview
This tutorial is divided into 5 parts; they are:

1. Normality Tests
 Shapiro-Wilk Test
 D’Agostino’s K^2 Test
 Anderson-Darling Test
2. Correlation Tests
 Pearson’s Correlation Coefficient
 Spearman’s Rank Correlation
 Kendall’s Rank Correlation
 Chi-Squared Test
3. Stationary Tests
- Augmented Dickey-Fuller
 Kwiatkowski-Phillips-Schmidt-Shin
4. Parametric Statistical Hypothesis Tests
Student’s t-test
Paired Student’s t-test
Analysis of Variance Test (ANOVA)
Repeated Measures ANOVA Test
5. Nonparametric Statistical Hypothesis Tests
Mann-Whitney U Test
Wilcoxon Signed-Rank Test
Kruskal-Wallis H Test
Friedman Test

"""

'''
1. Normality Tests
This section lists statistical tests that you can use to check if your data has a Gaussian distribution.

Shapiro-Wilk Test
Tests whether a data sample has a Gaussian distribution.

Assumptions

Observations in each sample are independent and identically distributed (iid).
Interpretation

H0: the sample has a Gaussian distribution.
H1: the sample does not have a Gaussian distribution.
'''
# Example of the Shapiro-Wilk Normality Test
from scipy.stats import shapiro
data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
stat, p = shapiro(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably Gaussian')
else:
	print('Probably not Gaussian')