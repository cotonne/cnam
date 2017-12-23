#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import tree
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from matplotlib import cm as colormap
import pydotplus
from mpl_toolkits.mplot3d import Axes3D

plot_step=0.2
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
var_per = 0.98
#using scipy package
clf=PCA(n_components=3)
X_train=diabetes.data
X_train=clf.fit_transform(X_train)
print(clf.explained_variance_)
print("_"*30)
print(clf.components_[1,:])
print("__"*30)




# Parametres utilises pour la cross-validation
tuned_parameters = {'max_depth': range(1, 10),
                    'min_samples_leaf': range(1, 10)}

# On peut avoir l'ensemble des parametres que l'on peut utiliser pour 
# le tuning des hyperparametres via :
# clf.get_params()
clf = GridSearchCV(DecisionTreeRegressor(min_samples_split = 4),
  tuned_parameters, 
  cv=5
  )
# Il essaie automatiquement toutes les valeurs
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.3)
clf.fit(X_train, y_train)

print(clf.best_params_)
# On affiche le meilleur modele
nx = 100
x_min, x_max = plt.xlim()
xx = np.linspace(x_min, x_max, nx)
#plt.plot(xx,clf.best_estimator_.predict(xx),color='black')
#print(clf.best_estimator_.predict(xx.reshape(-1,1)))
print("Nombre de modeles testes : ")
print(clf.n_splits_)
print("Score sur les valeurs de tests : ")
print(clf.score(X_test, y_test))

dot_data = tree.export_graphviz(clf.best_estimator_, out_file=None,
                          filled=True, rounded=True,
                          special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("diabete.png")

# On affiche le resultat pour l'ensemble des possibilites
max_depths = np.array(range(1,10))
min_sample_splits = np.array(range(2,11))
xx, yy = np.meshgrid(max_depths, min_sample_splits)
Z = clf.cv_results_['mean_test_score'].reshape(xx.shape)

# affichage sous forme de wireframe des resultats des modeles evalues
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, Z, cmap=colormap.coolwarm)
ax.set_xlabel("max_depth")
ax.set_ylabel("min_samples_leaf")
ax.set_zlabel("Score moyen")

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax = fig.add_subplot(111, projection='3d')
ax.scatter(diabetes.data[:,2], diabetes.data[:,8], diabetes.target, 
	cmap=colormap.coolwarm)
ax.set_xlabel("BMI")
ax.set_ylabel("S8")
ax.set_zlabel("Y")

f, axarr = plt.subplots(3, 3)
for index in range(0, 9):
	axarr[index/3, index % 3].scatter(diabetes.data[:,index], diabetes.target, s=50)
	axarr[index/3, index % 3].set_title("Feature nb {0}".format(index))
plt.show()
