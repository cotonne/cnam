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

filename = 'german.data.csv'
delimiter = ' '
data = []

continuous_values = pd.read_csv(filename, delimiter=delimiter, 
    names=['Duration in month', 'Credit amount', 'Installement rate', 'Present residence since', 'Age', 'Number of existing credits', 'Nb of liable people'],
    usecols=[1,4, 7, 10, 12, 15, 17],
    dtype='float')

discrete_values = pd.read_csv(filename,delimiter=delimiter,
    names=['Status of checking account', 'Credit history', 
         'Purpose', 'Savings account', 'Present employment since', 
         'Personal status and sex', 'Guarantors', 'Property',
         'Other installment plans', 'Housing', 
         'Job', 'Telephone', 'Foreigner', 'Credit'],
    usecols=[0,2,3,5,6,8,9,11,13,14,16,18,19,20],
    dtype='S4')

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

dummies = pd.get_dummies(discrete_values.ix[:,:'Foreigner'], 
    columns=discrete_values.columns.tolist(), drop_first = True)

data = pd.concat([continuous_values, dummies], axis=1)
minmax = StandardScaler()
transformedData = minmax.fit_transform(data)
pca = PCA()
x2d = pca.fit_transform(transformedData)

#####################################
## Graphique des variances cumulees de l'ACP
#####################################
cum_var_exp = np.cumsum(pca.explained_variance_ratio_)
with plt.style.context('seaborn-whitegrid'):
    plt.figure()
    plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(len(pca.explained_variance_ratio_)), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

pca = PCA(n_components=2)
x2d = pca.fit_transform(transformedData)
labels = map(lambda v: 'r' if v == '1' else 'b', y)
plt.figure()
plt.scatter(x2d[:,0], x2d[:,1], c=labels, s=7, edgecolors='none', alpha=1.0)

plt.show()