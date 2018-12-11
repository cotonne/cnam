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

sns.set(style="whitegrid", color_codes=True)

# Reading CSV
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

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.20, random_state=42)

cost = np.vectorize(lambda t: 1 if t == '1' else 5)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)  # , sample_weight=cost(y_train))
print "Score apprentissage = %f" % clf.score(X_train, y_train)
print "Score test = %f" % clf.score(X_test, y_test)

dot_data = tree.export_graphviz(clf, out_file=None,
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("tree-simple.png")

tuned_parameters = {'max_depth': range(1, 10)
    , 'min_samples_leaf': range(1, 10)
    , 'min_samples_split': range(2, 10)
    , 'max_features': ["auto", "log2", None]
                    }

clf = GridSearchCV(DecisionTreeClassifier(),
                   tuned_parameters,
                   cv=5,
                   n_jobs=-1,
                   verbose=True
                   )

clf.fit(X_train, y_train)
print "Score apprentissage = %f" % clf.best_score_
print "Score test = %f" % clf.best_estimator_.score(X_test, y_test)

dot_data = tree.export_graphviz(clf.best_estimator_, out_file=None,
                                filled=True, rounded=True,
                                special_characters=True,
                                feature_names=X.columns)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("tree.png")
