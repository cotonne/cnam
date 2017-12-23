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

discrete_values = pd.read_csv(filename,delimiter=delimiter,
    names=['Status of checking account', 'Credit history', 
         'Purpose', 'Savings account', 'Present employment since', 
         'Personal status and sex', 'Guarantors', 'Property',
         'Other installment plans', 'Housing', 
         'Job', 'Telephone', 'Foreigner', 'Credit'],
    usecols=[0,2,3,5,6,8,9,11,13,14,16,18,19,20],
    dtype='S4')
credit_values = discrete_values['Credit']
# discrete_values = discrete_values.drop(labels='Credit', axis=1)


##### X²
# The p-value of a feature selection score indicates the probability 
# that this score or a higher score would be obtained if this variable 
# showed no interaction with the target.
#  scores are better if greater, p-values are better if smaller (and losses are better if smaller)
# Another general statement: scores are better if greater, p-values are better if smaller (and losses are better if smaller)
values = continuous_values.columns.values
# On discretise les variables continues en plusieurs groupes
dummies = pd.DataFrame()
dummies['Duration in month'] = pd.qcut(continuous_values['Duration in month'], 7, labels=False)
dummies['Credit amount'] = pd.qcut(continuous_values['Credit amount'], 10, labels=False)
dummies['Age'] = pd.qcut(continuous_values['Age'], 10, labels=False)

X = pd.concat([discrete_values, 
  dummies,
  continuous_values['Present residence since'],
  continuous_values['Installement rate'],
  continuous_values['Number of existing credits'],
  continuous_values['Nb of liable people']], axis=1)

y = credit_values.as_matrix()

f, axarr = plt.subplots(4, 5)
values = X.drop(labels='Credit', axis=1).columns
for index in range(0, len(values)):
  print 'Reading ' + values[index]
  sns.countplot(x=values[index], hue="Credit", data=X, ax=axarr[index/5, index % 5])


## Stacked bar
f, axarr = plt.subplots(4, 5)
values = X.drop(labels='Credit', axis=1).columns
for index in range(0, len(values)):
  print 'Reading ' + values[index]
  ct = pd.crosstab(credit_values, X[values[index]], margins=True)
  # len - 1 parce que on retire la colonne ALL
  bar_width = 1
  totals = ct.values[-1][:-1]
  ok = np.array([float(i) / float(j) * 100 for  i,j in zip(ct.values[0][:-1], totals)])
  ko = np.array([float(i) / float(j) * 100 for  i,j in zip(ct.values[1][:-1], totals)])

  bar_l = np.arange (len(ct.values[0]) - 1)

  ax=axarr[index/5, index % 5]
  ax.bar(bar_l, ok,
     label='OK', alpha=0.9, color='#019600', width=bar_width, edgecolor='white')

  ax.bar(bar_l, ko, bottom=ok,
     label='KO', alpha=0.9, color='#3C5F5A', width=bar_width, edgecolor='white')

  # positions of the x-axis ticks (center of the bars as bar labels)
  #tick_pos = [i+(bar_width/2) for i in bar_l]
  # Set the ticks to be first names
  print ct.columns[:-1]
  ax.set_ylabel("Percentage")
  ax.set_xlabel(values[index])
  ax.set_xticks(bar_l + bar_width/2)
  ax.set_xticklabels(ct.columns[:-1].tolist())

  # Let the borders of the graphic
  # plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])
  # plt.ylim(-10, 110)

  # rotate axis labels
  # plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')


plt.show()