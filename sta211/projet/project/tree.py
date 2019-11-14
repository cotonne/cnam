# -*- coding: utf-8 -*-
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder

from util import codageDisjonctifComplet

filename = 'clean_data_train.csv'
delimiter = ';'
data = []

categories = {"gender": 'category', "copd": 'category',
              "previoushf": 'category', "afib": 'category', "cad": 'category',
              "hypertension": 'category'}
predictor = "lvefbin"
X = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False,
                dtype=dict(categories, **{predictor: 'category', "centre_country": 'category'}))
y = X[predictor].values.ravel()
X = X.drop(predictor, 1)

filename_test = 'clean_data_test.csv'
data_test = pd.read_csv(filename_test, header=0, sep=delimiter, error_bad_lines=False,
                        dtype=categories)

for i in categories.keys():
    label_encoder = LabelEncoder().fit(X[i])
    X[i] = label_encoder.transform(X[i])
    data_test[i] = label_encoder.transform(data_test[i])

X, data_test = codageDisjonctifComplet(X, data_test, "centre_country")

# Grid Search
tuned_parameters = {'n_estimators': range(100, 200, 10)
    , 'min_samples_leaf': range(4, 10, 2)
    , 'min_samples_split': range(15, 24, 2)
    , 'max_depth': range(10, 18, 2)
    , 'max_features': range(12, 22, 2)
                    }
clf = GridSearchCV(ExtraTreesClassifier(),
                   tuned_parameters,
                   cv=5,
                   n_jobs=-1,  # Use all processors
                   verbose=True
                   )
clf.fit(X, y)

print("Optimise ExtraTreesClassifier")
print("Score apprentissage  = %f" % clf.score(X, y))

print("Params")
print(clf.best_params_)
print("Best score")
print(clf.best_score_)
print("Variables")
print(X.columns)
print("Variable importance")
print(clf.best_estimator_.feature_importances_)

if len(data_test) != 987:
    sys.exit("Missing values")

pred_test = clf.best_estimator_.predict(data_test)
df = pd.DataFrame(pred_test)

import datetime

today = datetime.datetime.now()
df.to_csv(today.strftime('%Y%m%d%H%M') + "-python_extratrees.csv", index=False, encoding='utf-8', header=False)
