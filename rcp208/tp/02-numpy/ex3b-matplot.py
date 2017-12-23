# -*- coding: utf-8 -*-

import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
mammals = np.loadtxt('mammals.csv', delimiter=';', usecols=[2, 4, 8], skiprows=1)
print mammals[:5, :]
noms = np.genfromtxt('mammals.csv', dtype='str', delimiter=';', usecols=[0], skip_header=1)
print noms

from mpl_toolkits.mplot3d import Axes3D
fig, ax = plt.subplots()
ax = fig.add_subplot(111, projection='3d')
for i in range(0, len(noms), 2):
    x,y,z = mammals[i, 0], mammals[i, 1], mammals[i, 2]
    ax.scatter(x, y, z)
    ax.text(x, y, z, noms[i])


plt.show()
