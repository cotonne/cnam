from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import numpy as np
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
import pydotplus
from sklearn.model_selection import train_test_split
import sys
from joblib import Parallel, delayed
import multiprocessing
from matplotlib import cm as colormap
from mpl_toolkits.mplot3d import Axes3D

bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
digits = load_digits()

X=digits.data
y=digits.target

mult = 10
num_cores = 8

def evaluate_tree():
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)
	clf = tree.DecisionTreeClassifier()
	clf.fit(X_train, y_train)
	Z = clf.predict(X_test)
	return clf.score(X_test,y_test)

def evaluate_bagging(n_estimators = 200, max_samples=0.5, max_features=0.5):
	sys.stdout.write('.')
	sys.stdout.flush()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)
	clf = BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=max_samples, max_features=max_features, n_estimators=n_estimators)
	clf.fit(X_train, y_train)
	Z = clf.predict(X_test)
	return [max_samples, max_features, n_estimators, clf.score(X_test,y_test)]

# Calcul de la moyen avec un arbre classique
accurencies = [evaluate_tree() for i in range(0, 10 * mult)]
avg = np.average(accurencies)
std = np.std(accurencies)
print("Tree : Average Score = {0}, Std={1}".format(avg, std))

# Calcul avec le bagging en faisant varier le nombre d'arbres
accurencies_bagging =  np.array(Parallel(n_jobs=num_cores)(delayed(evaluate_bagging)() for i in range(0, 10 * mult)))
avg_bagging = np.average(accurencies_bagging[:,3])
std_bagging = np.std(accurencies_bagging[:,3])
print("Bagging : Score avg = {0}, std = {1}".format(avg_bagging, std_bagging))

# Idem en faisant varier le max_samples et le max_features
accurencies_samples_features = np.array(Parallel(n_jobs=num_cores)(delayed(evaluate_bagging)(200, i*0.1, j*0.1) for i in range(1, mult) for j in range(1, mult)))


# accurencies_bagging = [1 - evaluate_bagging(i) for i in range(2, 200, 2)]

#dot_data = tree.export_graphviz(clf, out_file=None,
#                          filled=True, rounded=True,
#                          special_characters=True)
#graph = pydotplus.graph_from_dot_data(dot_data)
#graph.write_png("digits.png")

# afficher une des images
f, axarr = plt.subplots(3, 3)
for index in range(0, 9):
	axarr[index/3, index % 3].matshow(digits.images[index])
	axarr[index/3, index % 3].set_title("Feature nb {0}".format(index))

f = plt.figure()
plt.scatter(accurencies_bagging[:, 2], accurencies_bagging[:, 3], s=50)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(accurencies_samples_features[:,0], accurencies_samples_features[:,1], accurencies_samples_features[:,3], cmap=colormap.coolwarm)
ax.set_xlabel("max_samples")
ax.set_ylabel("max_features")
ax.set_zlabel("Score")

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(accurencies_samples_features[:,0], accurencies_samples_features[:,1], accurencies_samples_features[:,3], cmap=colormap.coolwarm)
ax.set_xlabel("max_samples")
ax.set_ylabel("max_features")
ax.set_zlabel("Score")

plt.show()

