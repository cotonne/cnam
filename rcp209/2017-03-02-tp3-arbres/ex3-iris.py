#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from matplotlib import cm as colormap

# Paramètres
n_classes = 3
plot_colors = "bry" # blue-red-yellow
plot_step = 0.5

# Charger les données
iris = load_iris()

# Choisir les attributs longueur et largeur des pétales
pair = [1, 2, 3]

# Garder seulement les deux attributs
X = iris.data[:, pair]
y = iris.target
# Apprentissage de l'arbre
clf = DecisionTreeClassifier().fit(X, y)

# Prediction
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, plot_step), 
	np.arange(y_min, y_max, plot_step),
	np.arange(z_min, z_max, plot_step))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])

results = zip(xx.ravel(), yy.ravel(), zz.ravel(), Z.ravel())

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')

def plot(i, r, m):
	arr = np.array(filter(lambda x:x[3]==i, r))
	ax.scatter(arr[:, 0], arr[:, 1], zs=arr[:, 2], marker=m)

plot(0, results, 'o')
plot(1, results, 'v')
plot(2, results, 'v')
ax.set_xlabel(iris.feature_names[pair[0]])
ax.set_ylabel(iris.feature_names[pair[1]])
ax.set_zlabel(iris.feature_names[pair[2]])
plt.show()