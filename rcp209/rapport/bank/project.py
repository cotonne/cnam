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
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import tree

sns.set(style="whitegrid", color_codes=True)


# Reading CSV
#data = np.genfromtxt('full.csv',delimiter=' ')
filename = 'german.data.csv'
delimiter = ' '
data = []
with open(filename, 'rb') as csvfile:
  reader = csv.reader(csvfile, delimiter=delimiter)
  for row in reader:
    data.append(row)

# get frequency
transpose = map(list, zip(*data)) 
for row in transpose:
  counter = collections.Counter(row)
#  print "For :" + row.pop(0)
#  print counter.most_common(3)

# np.set_printoptions(threshold=np.nan)
continuous_values = np.genfromtxt(filename,delimiter=delimiter, skip_header=1, 
  	usecols=[4],
  	dtype='float')


values = continuous_values

f, axarr = plt.subplots(1, 1)
sns.distplot(values);

discrete_values = np.genfromtxt(filename,delimiter=delimiter, skip_header=1, 
    usecols=range(0,3) + range(5, 20),
    dtype='float')

values = discrete_values.columns.values
f, axarr = plt.subplots(4, 5)
for index in range(0, len(values)):
  print 'Reading ' + values[index]
  sns.countplot(x=values[index], data=discrete_values, 
    palette="Greens_d", ax=axarr[index/4, index % 5]);


plt.show()
