#!/usr/bin/python
# -*- coding: utf-8 

import numpy as np
import pylab as pl
from sklearn import svm

xx, yy = np.meshgrid(np.linspace(-7, 7, 500), np.linspace(-7, 7, 500))
X = 0.3 * np.random.randn(100, 2)
X = np.r_[X + 2, X - 2]

# Ajouter 10 % de 'outliers' (ce qui nous conduit à utiliser un nu=0.1)
X = np.r_[X, np.random.uniform(low=-6, high=6, size=(20, 2))]

# Construire le model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.01)
clf.fit(X)

# Afficher les points et les vecteurs les plus proches de du plan de séparation
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
y_pred = clf.predict(X)

pl.set_cmap(pl.cm.Paired)
pl.contourf(xx, yy, Z)
pl.scatter(X[y_pred>0,0], X[y_pred>0,1], c='white', label='inliers')
pl.scatter(X[y_pred<=0,0], X[y_pred<=0,1], c='black', label='outliers')
pl.axis('tight')
pl.legend()
pl.show()