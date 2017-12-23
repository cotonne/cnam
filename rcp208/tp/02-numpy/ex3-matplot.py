# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

# Graph simple
plt.plot(np.random.rand(100))
plt.show(block=False)

# Affichage d'un fichier
geyser = np.loadtxt('geyser.txt', dtype = float, delimiter = ' ')
nb_line, nb_col = geyser.shape

# Les colonnes sont centrées réduites
for i in range(0, nb_col):
    c = geyser[:, i]
    mean = c.mean()
    std = c.std()
    geyser[:, i] = (c - mean) / std

plt.plot(geyser)

# représentation des points en x, y
plt.figure()
plt.plot(geyser[:, 0] * 2.5, geyser[:, 1] , 'r+')

# affichage en 3D
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(np.random.randn(100), np.random.randn(100), np.random.randn(100), c=['r'])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')



plt.show()
