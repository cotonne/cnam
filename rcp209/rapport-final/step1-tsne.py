#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numpy import genfromtxt

import csv
import collections
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2 as chi2_s
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from matplotlib import cm as colormap
import pydotplus
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

sns.set(style="whitegrid", color_codes=True)

# higher is better
from sklearn.metrics import make_scorer
scores = [[0, 1], [5,0]]
def custom_scoring(y_true, y_pred, **kwargs):
  res = map(lambda yt, yp: scores[int(yt) - 1][int(yp) - 1], y_true, y_pred)
  return len(filter(lambda x: x == 0, res))/float(len(res))

scoring = make_scorer(custom_scoring)


# Reading CSV
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

#####################################
# Reprojection
#####################################
data = pd.concat([discrete_values.apply(LabelEncoder().fit_transform), continuous_values], axis=1)
X = data.as_matrix()
y = credit_values.as_matrix()
X_scaled = preprocessing.MinMaxScaler().fit_transform(X)

#####################################
## t-SNE sur deux composantes
#####################################
tsne = TSNE(n_components=2, random_state=0, init='pca', 
    perplexity=50, verbose=2, n_iter=2000, n_iter_without_progress=100,
    learning_rate=500)


x2d = tsne.fit_transform (X_scaled)


labels = map(lambda v: 'r' if v == '1' else 'b', y)
plt.figure()
plt.scatter(x2d[:,0], x2d[:,1], c=labels, s=7, edgecolors='none', alpha=1.0)

plt.show()

#####################################
## t-SNE sur trois composantes
#####################################
tsne = TSNE(n_components=3, random_state=0, init='pca', 
    perplexity=30, verbose=2, n_iter=2000, n_iter_without_progress=100,
    learning_rate=500)

x3d = tsne.fit_transform (X)

xx, yy, zz = x3d[:,0], x3d[:,1], x3d[:,2]
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(xx, yy, zz, c=labels)
plt.show()