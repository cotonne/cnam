from sklearn import manifold

from util import loadModel, load_data, convexHulls, best_ellipses, visualization

model = loadModel("model-ex2")
(X_train, Y_train, X_test, Y_test) = load_data()

# On veut maintenant extraire la couche cachée (donc un vecteur de dimension 100) pour chacune des images de la base de test.
# model.pop() supprime la couche au sommet du modèle
model.pop()
# model.pop() supprime la couche d’activation softmax
model.pop()
# Ensuite on peut appliquer la méthode model.predict(X_test) sur l’ensemble des données de test.
predict = model.predict(X_test)

tsne = manifold.TSNE(init='pca', perplexity=30, verbose=2)
x2d = tsne.fit_transform(predict)
convex_hulls = convexHulls(x2d, Y_test)
ellipses = best_ellipses(x2d, Y_test)
visualization(x2d, Y_test, convex_hulls, ellipses, "", 1)

