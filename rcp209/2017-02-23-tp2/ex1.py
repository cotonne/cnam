import numpy as np    # si pas encore fait
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut

# definir matrices de rotation et de dilatation
rot = np.array([[0.94, -0.34], [0.34, 0.94]])
sca = np.array([[3.4, 0], [0, 2]])

# generer donnees classe 1
np.random.seed(150)
c1d = (np.random.randn(400,2)).dot(sca).dot(rot)

# generer donnees classe 2
c2d1 = np.random.randn(100,2)+[-10, 2]
c2d2 = np.random.randn(100,2)+[-7, -2]
c2d3 = np.random.randn(100,2)+[-2, -6]
c2d4 = np.random.randn(100,2)+[5, -7]

data = np.concatenate((c1d, c2d1, c2d2, c2d3, c2d4))

# generer etiquettes de classe
l1c = np.ones(400, dtype=int)
l2c = np.zeros(400, dtype=int)
labels = np.concatenate((l1c, l2c))

# decoupage initial en donnees d'apprentissage et donnees de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.5)

fig = plt.figure()
cm = np.array(['r','g'])
plt.scatter(X_train[:,0],X_train[:,1],c=cm[y_train],s=50,edgecolors='none')
plt.scatter(X_test[:,0],X_test[:,1],c='none',s=50,edgecolors=cm[y_test])

# emploi de PMC
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1)

# KFold pour differentes valeurs de k
from sklearn.model_selection import KFold
# valeurs de k
kcvfs=np.array([2, 3, 5, 7, 10, 13, 16, 20])
# preparation des listes pour stocker les resultats
kcvscores = list()
kcvscores_std = list()

kcvscores_err = list()
kcvscores_std_err = list()



for kcvf in kcvfs:    # pour chaque valeur de k
  kf = KFold(n_splits=kcvf)
  these_scores = list()
  these_scores_err = list()
  # apprentissage puis evaluation d'un modele sur chaque split
  for train_idx, test_idx in kf.split(X_train):
    clf.fit(X_train[train_idx], y_train[train_idx])
    these_scores.append(clf.score(X_train[test_idx], y_train[test_idx]))
    these_scores_err.append(clf.score(X_test, y_test))
  # calcul de la moyenne et de l'ecart-type des performances obtenues
  kcvscores.append(np.mean(these_scores))
  kcvscores_std.append(np.std(these_scores))
  kcvscores_err.append(np.mean(these_scores_err))
  kcvscores_std_err.append(np.std(these_scores_err))

loo = LeaveOneOut()
clf.fit(X_train[train_idx], y_train[train_idx])
print("Score sur train = ")
print(clf.score(X_train[test_idx], y_train[test_idx]))
print("Score sur test  = ")
print(clf.score(X_test, y_test))

# creation de np.array a partir des listes
kcvscores, kcvscores_std = np.array(kcvscores), np.array(kcvscores_std)
kcvscores_err, kcvscores_std_err = np.array(kcvscores_err), np.array(kcvscores_std_err)

# affichage performance moyenne +- 1 ecart-type pour chaque k
fig = plt.figure()
plt.plot(kcvfs, kcvscores, 'b')
plt.plot(kcvfs, kcvscores+kcvscores_std, 'b--')
plt.plot(kcvfs, kcvscores-kcvscores_std, 'b--')

plt.plot(kcvfs, kcvscores_err, 'r')
plt.plot(kcvfs, kcvscores_err+kcvscores_std_err, 'r--')
plt.plot(kcvfs, kcvscores_err-kcvscores_std_err, 'r--')

plt.show()
