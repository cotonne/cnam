#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


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


# higher is better
from sklearn.metrics import make_scorer

scores = [[0, 1], [5, 0]]


def custom_scoring(y_true, y_pred, **kwargs):
    res = map(lambda yt, yp: scores[int(yt) - 1][int(yp) - 1], y_true, y_pred)
    return len(filter(lambda x: x == 0, res)) / float(len(res))


scoring = make_scorer(custom_scoring)

filename = 'data_train_ncp_15.csv'
delimiter = ';'
data = []

X = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False,
                usecols=["gender", "bmi", "age", "sbp", "hr", "hypertension", "previoushf", "cad"],
                dtype={
                    "gender": 'S4', "bmi": 'float', "age": 'float',
                    "sbp": 'float', "hr": 'float',
                    "hypertension": 'S4', "previoushf": 'S4', "cad": 'S4'})

Y_train = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False,
                      usecols=["lvefbin"],
                      dtype={"lvefbin": 'S4'})

X_test = pd.read_csv("data_test_ncp_15.csv", header=0, sep=";",
                     usecols=["gender", "bmi", "age", "sbp", "hr", "hypertension", "previoushf", "cad"],
                     dtype={
                         "gender": 'S4', "bmi": 'float', "age": 'float',
                         "sbp": 'float', "hr": 'float',
                         "hypertension": 'S4', "previoushf": 'S4', "cad": 'S4'})

#X = codageDisjonctifComplet(X, "centre")
#X = codageDisjonctifComplet(X, "country")

X_scaled = preprocessing.StandardScaler().fit_transform(X)
y = preprocessing.LabelEncoder().fit_transform(Y_train.values.ravel())
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)

tuned_parameters = {
    'kernel': ['rbf'], # 'linear', 'poly', 'rbf', 'poly'
    'degree': range(2, 5),
    'gamma': np.array(list(map(lambda l: 10 ** (-l), range(-4, 4))))
}

clf = GridSearchCV(
    svm.OneClassSVM(),
    tuned_parameters,
    # cv=5,
    n_jobs=4,
    verbose=2,
    # scoring=scoring
    scoring="accuracy"
)

clf.fit(X_train, y_train)

col = clf.best_estimator_.predict(X_train)
col_bis = (col >= 0).astype(int)
labels = np.array(['bad', 'good'])
col_bis = labels[col_bis]
sum = np.sum(col_bis == labels[y_train])/col_bis.shape[0]

print("No Score apprentissage = {}".format(sum))

col = clf.best_estimator_.predict(X_test)
col_bis = (col >= 0).astype(int)
labels = np.array(['bad', 'good'])
col_bis = labels[col_bis]
sum = np.sum(col_bis == labels[y_test])/col_bis.shape[0]
print("No Score test = {}".format(sum))

fig = plt.figure()
plt.plot(range(-5, 5), clf.cv_results_['mean_test_score'])
plt.xlabel('10^x')
plt.ylabel('Test score')
plt.show()

####
#
# pred_test = clf.best_estimator_.predict(data_test)
# print pred_test
# df = pd.DataFrame(pred_test)
# df.to_csv("python_svm.csv")

# cparams = np.array(range(-2, 2))
# kparams = np.array(['linear', 'poly', 'rbf', 'sigmoid'])
# gparams = np.array(range(-4, 4))
# xx, yy, zz= np.meshgrid(cparams, kparams, gparams)
#
# # affichage sous forme de wireframe des resultats des modeles evalues
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# Z = clf.cv_results_['mean_test_score'].reshape((len(xx), len(yy), len(zz)))
# #ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(xx, yy, Z, cmap=colormap.coolwarm)
# ax.set_xlabel("Profondeur")
# ax.set_ylabel("Nombre d'estimateurs")
# ax.set_zlabel("Score moyen")
# plt.show()
