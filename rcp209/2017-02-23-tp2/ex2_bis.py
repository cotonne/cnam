import numpy as np    # si pas encore fait
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from matplotlib import cm as colormap
from scipy.stats import randint as sp_randint
from scipy.stats import alpha as sp_alpha
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
from sklearn.model_selection import RandomizedSearchCV

# Parametres utilises pour la cross-validation
param_dist = {'hidden_layer_sizes': sp_randint(5, 500),
                    'alpha': sp_alpha(5)}

# On peut avoir l'ensemble des parametres que l'on peut utiliser pour 
# le tuning des hyperparametres via :
# clf.get_params()
clf = RandomizedSearchCV(MLPClassifier(solver='lbfgs', alpha=1),
  param_dist, 
  n_iter=24
  #,n_jobs=8
  )
# Il essaie automatiquement toutes les valeurs
clf.fit(X_train, y_train)

print(clf.best_params_)
# On affiche le meilleur modele
nx = 100
x_min, x_max = plt.xlim()
xx = np.linspace(x_min, x_max, nx)
#plt.plot(xx,clf.best_estimator_.predict(xx),color='black')
#print(clf.best_estimator_.predict(xx.reshape(-1,1)))
print("Nombre de modeles testes : ")
print(clf.n_splits_)
print("clf.cv_results_ = ")
print(clf.cv_results_)
print("Score sur les valeurs de tests : ")
print(clf.score(X_test, y_test))
# On affiche le resultat pour l'ensemble des possibilites
res = clf.cv_results_
alphas, hiddens = res['params']
n_hidden = np.array(hiddens)
alphas = np.array(alphas)
xx, yy = np.meshgrid(n_hidden, alphas)
Z = clf.cv_results_['mean_test_score'].reshape(xx.shape)

# affichage sous forme de wireframe des resultats des modeles evalues
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, Z, cmap=colormap.coolwarm)
ax.set_xlabel("Nombre de neurones caches")
ax.set_ylabel("Weight decay")
ax.set_zlabel("Score moyen")
plt.show()
