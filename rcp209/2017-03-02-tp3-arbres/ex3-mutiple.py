#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from matplotlib import cm as colormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

# Paramètres
n_classes = 3
plot_colors = "bry" # blue-red-yellow
plot_step = 0.1

# Charger les données
iris = load_iris()

# Choisir les attributs longueur et largeur des pétales

features_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

def evaluate(index, pair):
	# Garder seulement les deux attributs
	X = iris.data[:, pair]
	y = iris.target

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

	# Apprentissage de l'arbre
	clf = DecisionTreeClassifier().fit(X_train, y_train)

	# Evaluation de l'erreur
	learning_error = clf.score(X_train, y_train)
	test_error = clf.score(X_test, y_test)

	# Prediction
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	
	# Rendering
	print(str(pair[0]) + " - " + str(pair[1]))
	plt.subplot(3, 2, index + 1)
	cm = np.array(['r','g', 'b'])
	plt.scatter(xx, yy, edgecolors=cm[Z], c='none',  s=1)
	plt.scatter(X_train[:,0], X_train[:,1], c=cm[y_train],s=50,edgecolors='none')
	plt.scatter(X_test[:,0], X_test[:,1], c='none',s=50, edgecolors=cm[y_test])
	plt.xlabel(features_names[pair[0]])
	plt.ylabel(features_names[pair[1]])
	str_learn = "{0:.3f}".format(learning_error)
	str_error = "{0:.3f}".format(test_error)
	plt.title("Erreur d'apprentissage : " + str_learn + " - Erreur de test : " + str_error)

plt.figure(figsize=(3, 2))
combinaisons = [[i, j] for i in range(0, 4) for j in range(0, 4) if(i < j)]
for index in range(0, len(combinaisons)):
	pair = combinaisons[index]
	evaluate(index, pair)

plt.show()