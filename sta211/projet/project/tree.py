#!/usr/bin/python2
# -*- coding: utf-8 -*-
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from matplotlib import cm as colormap
from mpl_toolkits.mplot3d import axes3d, Axes3D

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

print len(data_test)

if len(data_test) != 987:
    sys.exit("Missing values")

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.20, random_state=42)

# Basic extra tree
clf = ExtraTreesClassifier(n_estimators=100)
clf.fit(X_train, y_train)
print "ExtraTreesClassifier"
print "Score apprentissage  = %f" % clf.score(X_train, y_train)
print "Score test = %f" % clf.score(X_test, y_test)

# Grid Search
tuned_parameters = {'n_estimators': range(50, 450, 50)
    , 'min_samples_leaf': range(1, 10, 2)
    , 'min_samples_split': range(2, 10, 2)
    , 'max_depth': range(1, 12, 2)
                    }

tuned_parameters = {  # 'n_estimators': range(450, 450, 50)
    'min_samples_leaf': range(1, 10, 2)
    , 'min_samples_split': range(2, 10, 2)
    , 'max_depth': range(1, 12, 2)
}

clf = GridSearchCV(ExtraTreesClassifier(max_depth=5),
                   tuned_parameters,
                   cv=5,
                   n_jobs=-1,  # Use all processors
                   verbose=True
                   )

clf.fit(X_train, y_train)
print "Optimise ExtraTreesClassifier"
print "Score apprentissage  = %f" % clf.score(X_train, y_train)
print "Score test = %f" % clf.score(X_test, y_test)

print "Params"
print clf.best_params_
print "Best score"
print clf.best_score_

pred_test = clf.best_estimator_.predict(data_test)
print pred_test
df = pd.DataFrame(pred_test)
df.to_csv("python_extratrees.csv")

max_depth = np.array(range(1, 12, 2))
min_samples_split = np.array([range(2, 10, 2)])
xx, yy = np.meshgrid(max_depth, min_samples_split)

# affichage sous forme de wireframe des resultats des modeles evalues
fig = plt.figure()
ax = fig.gca(projection='3d')
Z = clf.cv_results_['mean_test_score'].reshape(xx.shape)
# ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, Z, cmap=colormap.coolwarm)
ax.set_xlabel("Profondeur")
ax.set_ylabel("Nombre d'estimateurs")
ax.set_zlabel("Score moyen")
plt.show()

fig = plt.figure()
plt.plot(range(50, 1000, 50), clf.cv_results_['mean_test_score'])
plt.show()
