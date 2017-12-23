import numpy as np    # si pas encore fait
import matplotlib.pyplot as plt
from matplotlib import cm as colormap
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)


def gen(nbMin, depth):
  clf = tree.DecisionTreeClassifier(min_samples_leaf = nbMin, max_depth=depth)
  # clf = clf.fit(iris.data, iris.target)
  clf = clf.fit(X_train, y_train)

  with open("iris.dot", 'w') as f:
      f = tree.export_graphviz(clf, out_file=f)

  # Generation d'un pdf via : dot -Tpdf iris.dot -o iris.pdf
  import pydotplus
  dot_data = tree.export_graphviz(clf, out_file=None,
                              feature_names=iris.feature_names,
                              class_names=iris.target_names,
                              filled=True, rounded=True,
                              special_characters=True)
  graph = pydotplus.graph_from_dot_data(dot_data)
  graph.write_png("iris-" + str(nbMin) + "-" + str(depth) +".png")
  return clf.score(X_test, y_test)

# on regarde par rapport a la meilleure generalisation
results = [(i, j, gen(i, j)) for i in [1, 2, 6, 10] for j in [1, 2, 4, 10]]
print(sorted(results, key=lambda r:r[2]))


# On calcule de maniere exhaustive
tuned_parameters = {'min_samples_leaf':[1, 2, 6, 10],
                    'max_depth':   [1, 2, 4, 10]}


clf = GridSearchCV(tree.DecisionTreeClassifier(),
  tuned_parameters, 
  cv=5
  #,n_jobs=8
  )

# clf = clf.fit(iris.data, iris.target)
clf = clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))



import pydotplus
dot_data = tree.export_graphviz(clf.best_estimator_, out_file=None,
                          feature_names=iris.feature_names,
                          class_names=iris.target_names,
                          filled=True, rounded=True,
                          special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("iris.png")


# On affiche le resultat pour l'ensemble des possibilites
n_hidden = np.array([1, 2, 6, 10])
alphas = np.array([1, 2, 4, 10])
xx, yy = np.meshgrid(n_hidden, alphas)

# affichage sous forme de wireframe des resultats des modeles evalues
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
Z = clf.cv_results_['mean_test_score'].reshape(xx.shape)
#ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, Z, cmap=colormap.coolwarm)
ax.set_xlabel("Min sample leaf")
ax.set_ylabel("Max depth")
ax.set_zlabel("Score moyen")
plt.show()


