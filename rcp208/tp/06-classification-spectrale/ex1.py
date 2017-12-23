
# coding: utf-8

# # Classification spectrale de données générées

# In[1]:

import numpy as np
from numpy import newaxis

# data1 = np.hstack((np.random.randn(100, 3) + [3, 3, 3],    (np.ones(100))[:newaxis]))
data1 = np.hstack((np.random.randn(100,3) + [3,3,3],(np.ones(100))[:,newaxis]))
data2 = np.hstack((np.random.randn(100,3) + [-3,-3,-3],(2 * np.ones(100))[:,newaxis]))
data3 = np.hstack((np.random.randn(100,3) + [-3,3,3],(3 * np.ones(100))[:,newaxis]))
data4 = np.hstack((np.random.randn(100,3) + [-3,-3,3],(4 * np.ones(100))[:,newaxis]))
data5 = np.hstack((np.random.randn(100,3) + [3,3,-3],(5 * np.ones(100))[:,newaxis]))

data = np.concatenate((data1, data2, data3, data4, data5))
print(data.shape)
np.random.shuffle(data)


# Appliquez la classification spectrale avec une construction du graphe sur la base des k plus proches voisins 

# In[ ]:

from sklearn.cluster import SpectralClustering
spectral = SpectralClustering(n_clusters = 5, eigen_solver='arpack', 
                              affinity='nearest_neighbors', n_neighbors = 10).fit(data[:,:3])

# Visualisons les résultats de cette classification

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], c=spectral.labels_)


# L’indice de Rand ajusté permet d’évaluer la cohérence entre les groupes de départ et le partitionnement trouvé par classification automatique

# In[ ]:

from sklearn import metrics
print metrics.adjusted_rand_score(spectral.labels_, data[:,3])

# Répétez plusieurs fois la classification avec des valeurs différentes (mais ≥1≥1) 
# pour le paramètre n_neighbors (dont la valeur par défaut est 10). 
for i in range(2, 11):
	spectral = SpectralClustering(n_clusters = 5, eigen_solver='arpack', 
                              affinity='nearest_neighbors', n_neighbors = i).fit(data[:,:3])
	print i, " = ", metrics.adjusted_rand_score(spectral.labels_, data[:,3])
# Que constatez-vous ? Expliquez.
# On constate que l'indice augmente de 2 à 5, puis stagne après
# Les k plus proches voisins sont ceux du noeud. On obtient une cohérence
# forte à partir de 5 

# Changez le mode de calcul de la matrice de similarités avec affinity = 'rbf'
# faites varier le paramètre correspondant gamma
# visualisez les résultats.
# gamma = 1/sigma^2 => gamma grand => loi normale pointu => centre regroupé 
for i in range(-2, 3):
	spectral = SpectralClustering(n_clusters = 5, eigen_solver='arpack', 
		              affinity='rbf', gamma = 10 ** (-i)).fit(data[:,:3])
	print 10 ** (-i), " = ", metrics.adjusted_rand_score(spectral.labels_, data[:,3])
	#fig = plt.figure()
	#fig.suptitle(str(i))
	#ax = fig.add_subplot(111, projection='3d')
	#ax.scatter(data[:,0], data[:,1], data[:,2], c=spectral.labels_)

# Pas de différence? Indice == 1 ==> méthode + efficace?

# Question : (→→ Compte-rendu) 
# Sur les 500 données générées comme au TP précédent (sur K-means) 
# suivant une distribution uniforme dans [0,1)^3, 
# appliquez la classification spectrale avec toujours n_clusters=5 
# et visualisez les résultats. 
# Examinez la stabilité (en utilisant l’indice de Rand) des partitionnements obtenus. 
# Observez-vous des différences significatives par rapport aux résultats obtenus lors du TP 
# sur K-means ? Expliquez.
print "Compte-rendu"
d = np.random.uniform(0, 1, (500, 3))
print d.shape

max_k = 20
z = []
for k in range(0, max_k):
    z.append(SpectralClustering(n_clusters = 5, eigen_solver='arpack', 
		         affinity='nearest_neighbors').fit(d))

r = np.zeros((max_k, max_k)) 
for i in range(0, max_k):
    for j in range(0, max_k):
            r[i,j] = metrics.adjusted_rand_score(z[i].labels_, z[j].labels_)

from scipy import stats
print "MEAN = ", r.mean()
print "STD = ", r.std()
# MEAN =  0.995280561749
# STD =  0.00856658735177
# On trouve quasiment toujours la même répartition 
# la variation vient de K-Means à la fin 
# Cependant, la préparation est très efficace
 
fig = plt.figure()
fig.suptitle(str(i))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(d[:,0], d[:,1], d[:,2], c=spectral.labels_)


#
# Classification spectrale des données « textures »
#

texture = np.loadtxt('texture.dat')
texture = np.genfromtxt('texture.dat', usecols=range(41), delimiter=' ', filling_values=0.0)
print texture
print texture.shape
lcls = lcls = np.genfromtxt('texture.dat', dtype=int, usecols=(41), delimiter=' ', filling_values=0.0)
print "LCLS = ", lcls
print lcls.shape

np.random.shuffle(texture)
spectral = SpectralClustering(n_clusters=11, eigen_solver='arpack',
               affinity='nearest_neighbors').fit(texture[:,:40])
print metrics.adjusted_rand_score(spectral.labels_, texture[:,40])

# (→→ Compte-rendu) Après une analyse discriminante de ces données, 
# appliquez la classification spectrale avec n_clusters = 11 aux 
# données projetées dans l’espace discriminant. 
# Que constatez-vous ? 
# Expliquez. 
# Visualisez les résultats.
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()

lda.fit(texture, lcls)

print lda.explained_variance_ratio_

tlda = lda.transform(texture)

# plt.show()




