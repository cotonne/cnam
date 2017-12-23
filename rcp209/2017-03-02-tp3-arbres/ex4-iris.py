#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import tree
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


def calc(X, y, depth):
	regr_1 = DecisionTreeRegressor(max_depth=depth)
	regr_1.fit(X, y)
	X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
	y_1 = regr_1.predict(X_test)
	return X_test, y_1, depth

X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
# Créer les données d'apprentissage
y[::5] += 3 * (0.5 - rng.rand(16))

ranges = range(1, 10, 2)
results = [calc(X, y, depth) for depth in ranges]

# Affichage des résultats
plt.figure()
plt.scatter(X, y, c="darkorange", label="data")
colors = map(lambda x:str(float(x)/12), ranges)
for xt, yt, depth in results:
	plt.figure()
	plt.scatter(X, y, c="darkorange", label="data")
	plt.plot(xt, yt, color=colors[(depth - 1) / 2], label="max_depth=" + str(depth), linewidth=2)
	plt.xlabel("data")
	plt.ylabel("target")
	plt.title("Decision Tree Regression")
	plt.legend()
plt.show()
