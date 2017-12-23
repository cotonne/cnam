# encoding: utf8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
# Chargement des données
iris = datasets.load_iris()
# Garder juste les premiers deux attributs
X = iris.data[:, :2]
y = iris.target
# Pour afficher la surface de décision on va discrétiser l'espace avec un pas h
h = .02
C = 1 # paramètre de régularisation
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(X, y)

# créer la surface de décision discretisée
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
titles = ['SVC with linear kernel', 'LinearSVC (linear kernel)']

for i, clf in enumerate((svc, lin_svc)):
        plt.subplot(1, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        # Utiliser une palette de couleurs
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # Afficher aussi les points d'apprentissage
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.jet)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])
plt.show()