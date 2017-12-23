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

# higher is better
from sklearn.metrics import make_scorer
scores = [[0, 1], [5,0]]
def custom_scoring(y_true, y_pred, **kwargs):
  res = map(lambda yt, yp: scores[int(yt) - 1][int(yp) - 1], y_true, y_pred)
  return len(filter(lambda x: x == 0, res))/float(len(res))

scoring = make_scorer(custom_scoring)

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

# Basic extra tree
clf = ExtraTreesClassifier()
clf.fit(X_train, y_train)
print "ExtraTreesClassifier"
print "Score apprentissage  = %f" % clf.score(X_train, y_train)
print "Score test = %f" % clf.score(X_test, y_test)

# Grid Search
tuned_parameters = { 'n_estimators' : range(50, 450, 50)
                    , 'min_samples_leaf': range(1, 10, 2)
                    , 'min_samples_split': range(2, 10, 2)
                    ,'max_depth': range(1, 12, 2)
                    }

clf = GridSearchCV(ExtraTreesClassifier(max_depth=5),
  tuned_parameters,
  cv=5,
  n_jobs=-1,
  verbose=True,
  scoring = scoring
  )

clf.fit(X_train, y_train)
print "Optimise ExtraTreesClassifier"
print "Score apprentissage  = %f" % clf.score(X_train, y_train)
print "Score test = %f" % clf.score(X_test, y_test)

max_depth = np.array(range(1, 12, 2))
n_estimators = np.array([range(50, 450, 50)])
xx, yy = np.meshgrid(max_depth, n_estimators)

# affichage sous forme de wireframe des resultats des modeles evalues
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
Z = clf.cv_results_['mean_test_score'].reshape(xx.shape)
#ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, Z, cmap=colormap.coolwarm)
ax.set_xlabel("Profondeur")
ax.set_ylabel("Nombre d'estimateurs")
ax.set_zlabel("Score moyen")
plt.show()

fig = plt.figure()
plt.plot(range(50, 1000, 50),clf.cv_results_['mean_test_score'])
plt.show()

