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

filename = 'data_train.csv'
delimiter = ';'
data = []

X = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False,
                names=["id", "centre", "country", "gender", "bmi", "age", "egfr", "sbp", "dbp", "hr",
                       "lvef", "lvefbin", "copd", "hypertension", "previoushf", "afib", "cad"],
                usecols=["centre", "country", "gender", "bmi", "age", "egfr", "sbp", "dbp", "hr",
                         "copd", "hypertension", "previoushf", "afib", "cad"],
                dtype={
                    "centre": 'S4',
                    "country": 'S4', "gender": 'S4', "bmi": 'float', "age": 'float', "egfr": 'float',
                    "sbp": 'float', "dbp": 'float', "hr": 'float', "copd": 'S4',
                    "hypertension": 'S4', "previoushf": 'S4', "afib": 'S4', "cad": 'S4'})

y = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False,
                names=["id", "centre", "country", "gender", "bmi", "age", "egfr", "sbp", "dbp", "hr",
                       "lvef", "lvefbin", "copd", "hypertension", "previoushf", "afib", "cad"],
                usecols=["lvefbin"],
                dtype={"lvefbin": 'S4'})

data_test = pd.read_csv("data_test.csv", header=0, sep=";",
                        names=["id", "centre", "country", "gender", "bmi", "age", "egfr", "sbp", "dbp", "hr",
                               "copd", "hypertension", "previoushf", "afib", "cad"],
                        usecols=["centre", "country", "gender", "bmi", "age", "egfr", "sbp", "dbp", "hr",
                                 "copd", "hypertension", "previoushf", "afib", "cad"],
                        dtype={
                            "centre": 'S4',
                            "country": 'S4', "gender": 'S4', "bmi": 'float', "age": 'float', "egfr": 'float',
                            "sbp": 'float', "dbp": 'float', "hr": 'float', "copd": 'S4',
                            "hypertension": 'S4', "previoushf": 'S4', "afib": 'S4', "cad": 'S4'})

X_scaled = preprocessing.StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values.ravel(), test_size=0.20, random_state=42)

C = 1 # paramètre de régularisation
tuned_parameters = {'C': map(lambda l: 10 ** l, range(-5, 5))}

clf = GridSearchCV(
  svm.LinearSVC(),
  tuned_parameters,
  cv=5,
  n_jobs=4,
  verbose=True,
  # scoring=scoring
  )

clf.fit(X_train, y_train)
print "No Score apprentissage = {}".format(clf.best_estimator_.score(X_train, y_train))
print "No Score test = {}".format(clf.best_estimator_.score(X_test, y_test))

pred_test = clf.best_estimator_.predict(data_test)
print pred_test
df = pd.DataFrame(pred_test)
df.to_csv("python_linear-svc.csv")

fig = plt.figure()
plt.plot(range(-5, 5),clf.cv_results_['mean_test_score'])
plt.xlabel('10^x')
plt.ylabel('Test score')
plt.show()
