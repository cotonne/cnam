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
from sklearn import svm

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

# Reprojection
data = pd.concat([discrete_values.apply(LabelEncoder().fit_transform), continuous_values], axis=1)
X = data.as_matrix()
y = credit_values.as_matrix()

X_scaled = preprocessing.StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)

C = 1 # paramètre de régularisation
tuned_parameters = {'C': map(lambda l: 10 ** l, range(-5, 5))}

clf = GridSearchCV(
  svm.LinearSVC(),
  tuned_parameters,
  cv=5,
  n_jobs=8,
  verbose=True,
  scoring=scoring
  )

clf.fit(X_train, y_train)
print "No Score apprentissage = %f" % clf.best_estimator_.score(X_train, y_train)
print "No Score test = %f" % clf.best_estimator_.score(X_test, y_test)

fig = plt.figure()
plt.plot(range(-5, 5),clf.cv_results_['mean_test_score'])
plt.xlabel('10^x')
plt.ylabel('Test score')
plt.show()

#### SVC
tuned_parameters = {
  'C': map(lambda l: 10 ** (-l), range(-5, 5)),
  'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
  #'degree': range(2, 5),
  'gamma' : map(lambda l: 10 ** (-l), range(-5, 6))
  }

clf = GridSearchCV(
  svm.SVC(),
  tuned_parameters,
  cv=5,
  n_jobs=-1,
  verbose=True,
  scoring=scoring
  )

clf.fit(X_train, y_train)
print "No Score apprentissage = %f" % clf.best_estimator_.score(X_train, y_train)
print "No Score test = %f" % clf.best_estimator_.score(X_test, y_test)


cparams = np.array(range(-3, 5))
kparams = np.array(['linear', 'poly', 'rbf', 'sigmoid'])
gparams = np.array(range(0, 10))
xx, yy, zz= np.meshgrid(cparams, kparams, gparams)

# affichage sous forme de wireframe des resultats des modeles evalues
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
Z = clf.cv_results_['mean_test_score'].reshape((len(xx), len(yy), len(zz)))
#ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, Z, cmap=colormap.coolwarm)
ax.set_xlabel("Profondeur")
ax.set_ylabel("Nombre d'estimateurs")
ax.set_zlabel("Score moyen")
plt.show()