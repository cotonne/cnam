#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm as colormap
from sklearn import preprocessing
from sklearn import svm
# higher is better
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


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


filename = 'data_train_ncp_15.csv'
filename_test = "data_test_ncp_15.csv"
# filename = 'clean_data_train.csv'
# filename_test = "clean_data_test.csv"
delimiter = ';'

# id;centre;country;gender;copd;hypertension;previoushf;afib;cad;bmi;age;egfr;sbp;dbp;hr;lvefbin

data = []

X = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False,
                dtype={
                    "centre": 'S4',
                    "country": 'S4', "gender": 'S4', "bmi": 'float', "age": 'float', "egfr": 'float',
                    "sbp": 'float', "dbp": 'float', "hr": 'float', "copd": 'S4',
                    "hypertension": 'S4', "previoushf": 'S4', "afib": 'S4', "cad": 'S4'})
X = codageDisjonctifComplet(X, "centre")
X = codageDisjonctifComplet(X, "country")
X_scaled = preprocessing.StandardScaler().fit_transform(X)
X_train = X_scaled

y = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False,
                usecols=["lvefbin"],
                dtype={"lvefbin": 'S4'})
y = y.values.ravel()
labels, y = np.unique(y, return_inverse=True)
Y_train = y

X_test = pd.read_csv(filename_test, header=0, sep=";",
                     usecols=["gender", "bmi", "age", "egfr", "sbp", "dbp", "hr", "centre", "country",
                              "copd", "hypertension", "previoushf", "afib", "cad"],
                     dtype={
                         "centre": 'S4',
                         "country": 'S4', "gender": 'S4', "bmi": 'float', "age": 'float', "egfr": 'float',
                         "sbp": 'float', "dbp": 'float', "hr": 'float', "copd": 'S4',
                         "hypertension": 'S4', "previoushf": 'S4', "afib": 'S4', "cad": 'S4'})

X_test = codageDisjonctifComplet(X_test, "centre")
X_test = codageDisjonctifComplet(X_test, "country")
X_test_scaled = preprocessing.StandardScaler().fit_transform(X_test)
X_test = X_test_scaled

# X_train, X_test, Y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)

#### SVC
tuned_parameters = {
    'C': np.fromiter(map(lambda l: 10 ** (-l), range(-2, 2)), dtype=np.float64),
    'kernel': ['rbf', 'sigmoid'],  # 'linear', 'poly',
    'degree': range(2, 5),
    'gamma': np.fromiter(map(lambda l: 10 ** (-l), range(-4, 4)), dtype=np.float64)
}

clf = GridSearchCV(
    svm.SVC(),
    tuned_parameters,
    # cv=5,
    n_jobs=4,
    verbose=2
)

clf.fit(X_train, Y_train)
print(clf.best_estimator_)
print("No Score apprentissage = {}".format(clf.best_estimator_.score(X_train, Y_train)))
# print("No Score test = {}".format(clf.best_estimator_.score(X_test, y_test)))

pred_test = clf.best_estimator_.predict(X_test)
print(pred_test)
df = pd.DataFrame(labels[pred_test])

import datetime
today = datetime.datetime.now()
df.to_csv(today.strftime('%Y%m%d%H%M') + "-python_svm.csv", index=False, encoding='utf-8')
cparams = np.array(range(-2, 2))
kparams = np.array(['rbf', 'sigmoid'])
gparams = np.array(range(-4, 4))
xx, yy, zz = np.meshgrid(cparams, kparams, gparams)

# affichage sous forme de wireframe des resultats des modeles evalues
fig = plt.figure()
ax = fig.gca(projection='3d')
Z = clf.cv_results_['mean_test_score'].reshape((len(xx), len(yy), len(zz)))
# ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, Z, cmap=colormap.coolwarm)
ax.set_xlabel("Profondeur")
ax.set_ylabel("Nombre d'estimateurs")
ax.set_zlabel("Score moyen")
plt.show()
