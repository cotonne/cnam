#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numpy import genfromtxt

import csv
import collections
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2 as chi2_s
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn import tree

# Reading CSV
filename = 'german.data.csv'
delimiter = ' '
data = []

continuous_values = pd.read_csv(filename, delimiter=delimiter, 
    names=['Duration in month', 'Credit amount', 'Installement rate', 'Present residence since', 'Age', 'Number of existing credits', 'Nb of liable people'],
    usecols=[1,4, 7, 10, 12, 15, 17],
    dtype='float')


values = continuous_values.columns.values


sns.set(style="whitegrid", color_codes=True)
# Decile
print np.percentile(continuous_values, np.arange(0, 100, 10), axis=0)

# Affiche les valeurs sous forme de distribution
f, axarr = plt.subplots(4, 2)
for index in range(0, len(values)):
  print 'Reading ' + values[index]
  sns.distplot(continuous_values[values[index]], ax=axarr[index/2, index % 2])

f, axarr = plt.subplots(4, 2)
for index in range(0, len(values)):
  print 'Reading ' + values[index]
  sns.boxplot(continuous_values[values[index]], ax=axarr[index/2, index % 2])




discrete_values = pd.read_csv(filename,delimiter=delimiter,
    names=['Status of checking account', 'Credit history', 
         'Purpose', 'Savings account', 'Present employment since', 
         'Personal status and sex', 'Guarantors', 'Property',
         'Other installment plans', 'Housing', 
         'Job', 'Telephone', 'Foreigner', 'Credit'],
    usecols=[0,2,3,5,6,8,9,11,13,14,16,18,19,20],
    dtype='S4')
credit_values = discrete_values['Credit']
discrete_values = discrete_values.drop(labels='Credit', axis=1)




#####
# Calcul des tables de contingences
for index in discrete_values.columns:
  print pd.crosstab(credit_values, discrete_values[index], margins=True)

unique, counts = np.unique(discrete_values, return_counts=True)
dict(zip(unique, counts))

values = discrete_values.columns.values

# Affiche les valeurs sous forme de distribution
f, axarr = plt.subplots(5, 3, figsize=(10, 10))
for index in range(0, len(values)):
  print 'Reading ' + values[index]
  sns.countplot(x=values[index], data=discrete_values, 
    ax=axarr[index/3, index % 3]);
plt.savefig("continuous1.png")

with_credit = discrete_values.copy()
with_credit["Credit"] = credit_values
f, axarr = plt.subplots(5, 3, figsize=(10, 10))
for index in range(0, len(values)):
  print 'Reading ' + values[index]
  ax = sns.countplot(x=values[index], hue='Credit', data=with_credit,
    ax=axarr[index/3, index % 3]);

plt.savefig("continuous2.png")


#####################################
## Matrice des correlations
#####################################
continuous_values = pd.read_csv(filename, delimiter=delimiter, 
    names=['Duration in month'],
    usecols=[1],
    dtype='float')

discrete_values = pd.read_csv(filename,delimiter=delimiter,
    names=['Status of checking account', 'Credit history', 
         'Savings account', 'Present employment since', 
         'Property', 'Foreigner', 'Credit'],
    usecols=[0,2,5,6,11,19,20],
    dtype='S4')

credit_values = discrete_values['Credit']
discrete_values = discrete_values.drop(labels='Credit', axis=1)

dummies = pd.get_dummies(discrete_values.ix[:,:'Foreigner'], columns=discrete_values.columns.values, drop_first = True)


data = pd.concat([continuous_values, dummies], axis=1)

covariance_matrix = np.corrcoef(data.transpose())

#####################################
### Generation de la matrice des correlations sous forme de heatmap
#####################################
plt.figure(figsize=(16, 16), dpi=80)
df = data.corr()
labels = df.where(np.triu(np.ones(df.shape)).astype(np.bool))
labels = labels.round(2)
labels = labels.replace(np.nan,' ', regex=True)
mask = np.triu(np.ones(df.shape)).astype(np.bool)
ax = sns.heatmap(df, mask=mask, cmap='RdYlGn_r', fmt='', square=True, linewidths=1.5)
mask = np.ones((20, 20))-mask
ax = sns.heatmap(df, mask=mask, cmap=ListedColormap(['white']),annot=labels,cbar=False, fmt='', linewidths=1.5)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("correlation.png")
plt.matshow(covariance_matrix, cmap=plt.cm.gray)