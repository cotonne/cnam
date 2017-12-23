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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from matplotlib import cm as colormap
import pydotplus
from mpl_toolkits.mplot3d import Axes3D


sns.set(style="whitegrid", color_codes=True)


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

# Reprojection
data = pd.concat([discrete_values.apply(LabelEncoder().fit_transform), continuous_values], axis=1)
X = data.as_matrix()
y = credit_values.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



clf = AdaBoostClassifier(
                base_estimator=tree.DecisionTreeClassifier(max_depth=10),
                n_estimators=200,
                learning_rate=2)

clf.fit(X_train, y_train)
print "AdaBoostClassifier"
print "Score apprentissage = %f" % clf.score(X_train, y_train)
print "Score test = %f" % clf.score(X_test, y_test)

tuned_parameters = {'n_estimators': range(10, 400, 20)
                    , 'base_estimator': sum(map(lambda n:
                                        [DecisionTreeClassifier(max_depth=n, max_features="auto"),
                                        DecisionTreeClassifier(max_depth=n, max_features="log2"),
                                        DecisionTreeClassifier(max_depth=n)], range(2,10)), [])
                    }

clf = GridSearchCV(AdaBoostClassifier(),
  tuned_parameters,
  cv=5,
  n_jobs=-1,
  verbose=True
  )

clf.fit(X_train, y_train)
print "CV AdaBoostClassifier"
print "Score apprentissage = %f" % clf.best_score_
print "Score test = %f" % clf.best_estimator_.score(X_test, y_test)


clf = GradientBoostingClassifier(n_estimators = 20)
clf.fit(X_train, y_train)
print "GradientBoostingClassifier"
print "Score apprentissage = %f" % clf.score(X_train, y_train)
print "Score test = %f" % clf.score(X_test, y_test)

