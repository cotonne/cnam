#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from matplotlib import cm as colormap
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# higher is better
from sklearn.metrics import make_scorer
scores = [[0, 1], [5,0]]
def custom_scoring(y_true, y_pred, **kwargs):
  res = map(lambda yt, yp: scores[int(yt) - 1][int(yp) - 1], y_true, y_pred)
  return len(filter(lambda x: x == 0, res))/float(len(res))

def codageDisjonctifComplet(X, name):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    values = X[name].values.reshape(-1, 1)
    le = LabelEncoder()
    oneHotEncodedValues = le.fit_transform(values).reshape(-1, 1)
    onehotencoder = OneHotEncoder()
    features = pd.DataFrame(onehotencoder.fit_transform(oneHotEncodedValues).toarray())
    features.columns = onehotencoder.get_feature_names()
    X = X.drop(columns=[name])
    return X.join(features, lsuffix=name)

scoring = make_scorer(custom_scoring)
filename = 'data_train_ncp_15.csv'
delimiter = ';'

# id;centre;country;gender;copd;hypertension;previoushf;afib;cad;bmi;age;egfr;sbp;dbp;hr;lvefbin

data = []

X = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False,
                usecols=["centre", "country", "gender", "bmi", "age", "egfr", "sbp", "dbp", "hr",
                         "copd", "hypertension", "previoushf", "afib", "cad"],
                dtype={
                    "centre": 'S4',
                    "country": 'S4', "gender": 'S4', "bmi": 'float', "age": 'float', "egfr": 'float',
                    "sbp": 'float', "dbp": 'float', "hr": 'float', "copd": 'S4',
                    "hypertension": 'S4', "previoushf": 'S4', "afib": 'S4', "cad": 'S4'})

Y_train = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False,
                      usecols=["lvefbin"],
                      dtype={"lvefbin": 'S4'})
X_test = pd.read_csv("data_test_ncp_15.csv", header=0, sep=";",
                     usecols=["centre", "country", "gender", "bmi", "age", "egfr", "sbp", "dbp", "hr",
                              "copd", "hypertension", "previoushf", "afib", "cad"],
                     dtype={
                         "centre": 'S4',
                         "country": 'S4', "gender": 'S4', "bmi": 'float', "age": 'float', "egfr": 'float',
                         "sbp": 'float', "dbp": 'float', "hr": 'float', "copd": 'S4',
                         "hypertension": 'S4', "previoushf": 'S4', "afib": 'S4', "cad": 'S4'})


X = codageDisjonctifComplet(X, "centre")
X = codageDisjonctifComplet(X, "country")

X_scaled = preprocessing.StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values.ravel(), test_size=0.20, random_state=42)

#### SVC
tuned_parameters = {
  'C': map(lambda l: 10 ** (-l), range(-2, 2)),
  'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
  #'degree': range(2, 5),
  'gamma' : map(lambda l: 10 ** (-l), range(-4, 4))
  }

clf = GridSearchCV(
  svm.SVC(),
  tuned_parameters,
  #cv=5,
  n_jobs=4,
  verbose=2,
  # scoring=scoring
  )

clf.fit(X_train, y_train)
print("No Score apprentissage = {}".format(clf.best_estimator_.score(X_train, y_train)))
print("No Score test = {}".format(clf.best_estimator_.score(X_test, y_test)))

pred_test = clf.best_estimator_.predict(X_test)
print(pred_test)
df = pd.DataFrame(pred_test)
df.to_csv("python_svm.csv")

cparams = np.array(range(-2, 2))
kparams = np.array(['linear', 'poly', 'rbf', 'sigmoid'])
gparams = np.array(range(-4, 4))
xx, yy, zz= np.meshgrid(cparams, kparams, gparams)

# affichage sous forme de wireframe des resultats des modeles evalues
fig = plt.figure()
ax = fig.gca(projection='3d')
Z = clf.cv_results_['mean_test_score'].reshape((len(xx), len(yy), len(zz)))
#ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, Z, cmap=colormap.coolwarm)
ax.set_xlabel("Profondeur")
ax.set_ylabel("Nombre d'estimateurs")
ax.set_zlabel("Score moyen")
plt.show()