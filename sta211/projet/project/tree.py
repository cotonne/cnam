#!/usr/bin/python2
# -*- coding: utf-8 -*-
import sys

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder

filename = 'clean_data_train.csv'
delimiter = ';'
data = []

categories = {"gender": 'category', "copd": 'category',
              "previoushf": 'category', "afib": 'category', "cad": 'category',
              "centre_country": 'category'}
X = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False,
                dtype=dict(categories, **{"lvefbin": 'category'}))
y = X["lvefbin"]
X = X.drop("lvefbin", 1)

for i in categories.keys():
    X[i] = LabelEncoder().fit_transform(X[i])

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.20, random_state=42)

# Basic extra tree
clf = ExtraTreesClassifier(n_estimators=100, max_depth=5)
clf.fit(X_train, y_train)
print("ExtraTreesClassifier")
print("Score apprentissage  = %f" % clf.score(X_train, y_train))
print("Score test = %f" % clf.score(X_test, y_test))

# Grid Search
tuned_parameters = {'n_estimators': range(50, 450, 50)
    , 'min_samples_leaf': range(4, 20, 2)
    , 'min_samples_split': range(4, 20, 2)
    , 'max_depth': range(1, 12, 2)
                    }

clf = GridSearchCV(ExtraTreesClassifier(),
                   tuned_parameters,
                   cv=5,
                   n_jobs=-1,  # Use all processors
                   verbose=True
                   )
clf.fit(X_train, y_train)

print("Optimise ExtraTreesClassifier")
print("Score apprentissage  = %f" % clf.score(X_train, y_train))
print("Score test = %f" % clf.score(X_test, y_test))

print("Params")
print(clf.best_params_)
print("Best score")
print(clf.best_score_)
print("Variable importance")
print(clf.best_estimator_.feature_importances_)

filename_test = 'clean_data_test.csv'
data_test  = pd.read_csv(filename_test, header=0, sep=delimiter, error_bad_lines=False,
                dtype=categories)
for i in categories.keys():
    data_test[i] = LabelEncoder().fit_transform(data_test[i])

if len(data_test) != 987:
    sys.exit("Missing values")

pred_test = clf.best_estimator_.predict(data_test)
print(pred_test)
df = pd.DataFrame(pred_test)

import datetime
today = datetime.datetime.now()
df.to_csv(today.strftime('%Y%m%d%H%M') + "-python_extratrees.csv", index=False, encoding='utf-8')
