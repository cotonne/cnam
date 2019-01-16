#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import svm
# higher is better
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


def codageDisjonctifComplet(X, X_test, name):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    values = X[name].values
    le = LabelEncoder()
    oneHotEncodedValues = le.fit_transform(values).reshape(-1, 1)
    onehotencoder = OneHotEncoder()
    features = pd.DataFrame(onehotencoder.fit_transform(oneHotEncodedValues).toarray())
    features.columns = onehotencoder.get_feature_names()
    X = X.drop(columns=[name])
    X_test = X_test.drop(columns=[name])
    return X.join(features, lsuffix=name), X_test.join(features, lsuffix=name)


filename = 'clean_data_train.csv'
filename_test = 'clean_data_test.csv'
delimiter = ';'
data = []

categories = {"gender": 'category', "copd": 'category',
              "previoushf": 'category', "afib": 'category', "cad": 'category'}
X = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False,
                dtype=dict(categories, **{"lvefbin": 'category', "centre_country": 'category'}))
y = X["lvefbin"].values.ravel()
X = X.drop("lvefbin", 1)

X_test = pd.read_csv(filename_test, header=0, sep=delimiter, error_bad_lines=False,
                     dtype=dict(categories, **{"centre_country": 'category'}))

for i in categories.keys():
    label_encoder = LabelEncoder().fit(X[i])
    X[i] = label_encoder.transform(X[i])
    X_test[i] = label_encoder.transform(X_test[i])

X, X_test = codageDisjonctifComplet(X, X_test, "centre_country")

labels, y = np.unique(y, return_inverse=True)

# X_scaled = preprocessing.StandardScaler().fit_transform(X)
# X_train = X_scaled
# X_test_scaled = preprocessing.StandardScaler().fit_transform(X_test)
# X_test = X_test_scaled

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

clf.fit(X, y)
print(clf.best_estimator_)
print("No Score apprentissage = {}".format(clf.best_estimator_.score(X, y)))

pred_test = clf.best_estimator_.predict(X_test)
print(pred_test)
df = pd.DataFrame(labels[pred_test])

import datetime

today = datetime.datetime.now()
df.to_csv(today.strftime('%Y%m%d%H%M') + "-python_svm.csv", index=False, encoding='utf-8', header=False)

# cparams = np.array(range(-2, 2))
# kparams = np.array(['rbf', 'sigmoid'])
# gparams = np.array(range(-4, 4))
# xx, yy, zz = np.meshgrid(cparams, kparams, gparams)
#
# # affichage sous forme de wireframe des resultats des modeles evalues
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# Z = clf.cv_results_['mean_test_score'].reshape((len(xx), len(yy), len(zz)))
# # ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(xx, yy, Z, cmap=colormap.coolwarm)
# ax.set_xlabel("Profondeur")
# ax.set_ylabel("Nombre d'estimateurs")
# ax.set_zlabel("Score moyen")
# plt.show()
