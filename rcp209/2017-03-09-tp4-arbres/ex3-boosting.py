# http://cedric.cnam.fr/vertigo/Cours/ml2/tpForetsAleatoires.html#boosting
import numpy as np

from sklearn.datasets import load_digits
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from matplotlib import cm as colormap
from mpl_toolkits.mplot3d import Axes3D

digits = load_digits()
X=digits.data
y=digits.target


def evaluate(max_depth=5):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)
	clf = AdaBoostClassifier(
		base_estimator=tree.DecisionTreeClassifier(max_depth=max_depth), 
		n_estimators=200, 
		learning_rate=2)
	clf.fit(X_train, y_train)
	Z = clf.predict(X_test)
	return clf.score(X_test,y_test)

x = range(1, 12)
scores = [evaluate(depth) for depth in x]


plt.figure()
plt.plot(x, scores)
plt.xlabel("Profondeur de l'arbre")
plt.ylabel("Score")
plt.title("Score en fonction de la profondeur de l'arbre")


learning_rate_range = range(2, 10)
n_estimators_range  = range(1, 201, 10)
tuned_parameters = {'learning_rate': learning_rate_range,
                    'n_estimators' : n_estimators_range}

clf = GridSearchCV(
	AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=5)),
	tuned_parameters, 
	n_jobs=8,
	cv=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)
clf.fit(X_train, y_train)

xx = np.array(learning_rate_range)
yy = np.array(n_estimators_range)
xx, yy = np.meshgrid(xx, yy)
ZZ = clf.cv_results_['mean_test_score'].reshape(xx.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_wireframe(xx, yy, ZZ, cmap=colormap.coolwarm)
ax.set_xlabel("learning_rate")
ax.set_ylabel("n_estimators")
ax.set_zlabel("zz")

plt.show()

