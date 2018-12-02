from sklearn import manifold

from keras.datasets import mnist

from util import convexHulls, best_ellipses, visualization

(X, labels), (a, b) = mnist.load_data()
X = X.reshape(60000, 784)
X = X[:100,]
labels = labels[:100,]
X = X.astype('float32')
X /= 255

# Enveloppe convexe des points projetés pour chacune des classes
# ellipse de meilleure approximation des points
# Neighborhood Hit:
#   Pour chaque point, la métrique NH consiste à calculer,
#   pour les k plus proches voisins (k-nn) de ce point,
#   le taux de voisins qui sont de la même classe que le point considéré.
#   La métrique NH est ensuite moyennée sur l’ensemble de la base.

# En quoi les trois métriques ci-dessus sont-elles liées au problème de la séparabilité des classes ?
# Qu’est-ce qui les diffère ?
# enveloppe : tous les points ==> si séparer, tous les points sont séparés
# ellipse de meilleure approximation des points : ellipse qui contient 95% des points ==> si séparer, bonnes séparation des classes en moyennes
# NH : + NH est grand, plus un point est proche de ces voisins => bonne concentration
# Lien entre convex hull/enveloppe convexe & ellipse (convexe hull incluse dans ellipse)?
#

tsne = manifold.TSNE(init='pca', perplexity=30, verbose=2)
x2d = tsne.fit_transform(X)
convex_hulls = convexHulls(x2d, labels)
ellipses = best_ellipses(x2d, labels)
visualization(x2d, labels, convex_hulls, ellipses, "", 1)
