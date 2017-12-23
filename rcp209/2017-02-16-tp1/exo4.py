#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


# définir matrices de rotation et de dilatation
rot = np.array([[0.94, 0.34], [-0.34, 0.94]])
et2 = 3
sca = np.array([[10, 0], [0, et2]])
# générer données classe 1
np.random.seed(60)
datar = (np.random.randn(60,2)).dot(sca).dot(rot)


X_train1, X_test1, y_train1, y_test1 = train_test_split(datar[:,0], datar[:,1], test_size=0.33)
plt.scatter(X_train1,y_train1,s=50,edgecolors='none')
plt.scatter(X_test1,y_test1,c='none',s=50,edgecolors='blue')
x_min, x_max = plt.xlim()
nx = 100
xx = np.linspace(x_min, x_max, nx)

colors=['black', 'yellow', 'magenta', 'cyan', 'red', 'blue']

for exposant in range(0, 6):
  X_train1, X_test1, y_train1, y_test1 = train_test_split(datar[:,0], datar[:,1], test_size=0.33)
  clf = MLPRegressor(solver='lbfgs', alpha=10**(-exposant))
  # évaluation et affichage sur split1
  clf.fit(X_train1.reshape(-1,1), y_train1)
  clf.score(X_train1.reshape(-1,1), y_train1)
  clf.score(X_test1.reshape(-1,1), y_test1)
  plt.plot(xx,clf.predict(xx.reshape(-1,1)),color=colors[exposant])

plt.show()
