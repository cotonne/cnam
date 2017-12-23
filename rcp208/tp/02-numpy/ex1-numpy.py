# -*- coding: utf-8 -*-

import numpy as np


# Vecteur uni ou muldimensionnel composé d'éléments du même type
ti = np.array([1, 2, 3, 4])
print ti
print ti.dtype


tf = np.array([1.5, 2.5, 3.5, 4.5])
print tf.dtype

tf2d = np.array([[1.5, 2, 3], [4, 5, 6]])
print tf2d

tff = np.array([[ 1, 2, 3], [4, 5, 6]], dtype = float)
print tff

tfi = np.array([[1.5, 2, 3], [4, 5, 6]], dtype = int)
print tfi
print tfi.dtype
print tfi.shape
print tfi.ndim
print tfi.size

tz2d = np.zeros((3, 4))
print tz2d

tu2d = np.ones((3, 4))
print tu2d

# eq identity
id2d = np.eye(5)
print id2d

# créer un tableau sans initialisé les valeurs => aléatoire
tni2d = np.empty((3, 4))
print tni2d

ts1d = np.arange(0, 40, 5)
print ts1d

ts1d2 = np.linspace(0, 35, 8)
print ts1d2

# populate with random values from an uniform distribution
ta2d = np.random.rand(3, 5)
print ta2d

# génération avec une loi normale (standard normal distribution
tn2d = np.random.randn(3, 5)
print tn2d

# redimensionnement des données => reshape
tr = np.arange(20)
print tr

print tr.reshape(4,  5)
print tr.reshape(2, 10)
print tr.reshape(20)

# Au-delà de 3 dim
print np.random.rand(2, 3, 4)

# DIM 1 - Accès à une composante
# - tout du début jusqu'au 6è élément inclus
print tr[:6]

# - tout du début jusqu'au 10è, 1 sur 3
print tr[:10:3]

# modification des deux premiers
tr[:2] = 100
print tr

# DIM 2 - 
print ta2d

# - premier de la ligne 1 et de la colonne 1
print ta2d[0, 0]

# preimère ligne
print ta2d[0, :]

# premièere colonne
print ta2d[:, 0]

#  Sous-ens de la 1è à la 3è ligne exclus, puis de la 2è à la 4è ligne exclu
print ta2d[1:3, 2:4]

# Colonne impaire
print ta2d[:, 1::2]

for row in ta2d:
	print row

# Lecture / Ecriture tableau
tab = np.loadtxt('geyser.txt', dtype= float, delimiter=' ')
# autre solution : genfromtxt
print tab.shape
# pour sauvegarder : savetxt, savez, savez_compressed, ...

# Opérations simples sur tableaux
tb2d = tu2d * 2
print tb2d

print np.concatenate((tu2d, tb2d))
print np.concatenate((tu2d, tb2d), axis = 0) # ou np.stack
print np.concatenate((tu2d, tb2d), axis = 1) # ou np.hstack

# Ajouter un tableau unidimensionnel comme colonne à un tableau bidimensionnel
from numpy import newaxis
tu1d = np.ones(2)
print tu1d
tb2d = np.ones((2,2))*2
print tb2d
print tu1d[:, newaxis]
print np.column_stack((tu1d[:, newaxis], tb2d))
print np.hstack((tu1d[:,newaxis],tb2d))

# operation élément par élément
tsomme = tb2d - tu1d
print tsomme

print np.hstack((tb2d*2, tu1d[:, newaxis])) > 1
tb2d *= 3 # +=
print tb2d
print tb2d.sum()

print tab.min()
# max sur les lignes par colonne
print tab.max(axis=0)
print tab.sum(axis=0)

# Algère linéaire
g0 = tab[:2, :]
print g0
print g0.transpose()
# Inverse matrice
inv = np.linalg.inv(g0)
print inv
# multiplication matricielle
print g0.dot(inv)
print g0.trace()

#résolution de système linéaire
y = np.array([[5.], [7.]])
print np.linalg.solve(g0, y)

# valeurs et Vecteurs propres
valp, vecp = np.linalg.eig(g0)
print valp[0]
X = vecp[:,0]
# Verification des propriétés
print g0.dot(X)
print valp[0] * X


