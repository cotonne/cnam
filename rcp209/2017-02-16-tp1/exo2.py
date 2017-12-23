import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5)

# definir matrices de rotation et de dilatation
rot = np.array([[0.94, -0.34], [0.34, 0.94]])
sca = np.array([[3.4, 0], [0, 2]])
# generer donnees classe 1
np.random.seed(150)
c1d = (np.random.randn(100,2)).dot(sca).dot(rot)

# generer donnees classe 2
c2d1 = np.random.randn(25,2)+[-10, 2]
c2d2 = np.random.randn(25,2)+[-7, -2]
c2d3 = np.random.randn(25,2)+[-2, -6]
c2d4 = np.random.randn(25,2)+[5, -7]

data = np.concatenate((c1d, c2d1, c2d2, c2d3, c2d4))

# generer etiquettes de classe
l1c = np.ones(100, dtype=int)
l2c = np.zeros(100, dtype=int)
labels = np.concatenate((l1c, l2c))

cm = np.array(['r','g'])

nx, ny = 200, 200
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),np.linspace(y_min, y_max, ny))


# Split the data in two subsets : one for the training, one for the test
for size in np.arange(0.1, 0.9, 0.1):
    X_train1, X_test1, y_train1, y_test1 = train_test_split(data, labels, test_size=size)
    # evaluation et affichage sur split1
    clf.fit(X_train1, y_train1)
    print(str(size) + ", Apprentissage = " + str(clf.score(X_train1, y_train1)) + ", Test = " + str(clf.score(X_test1, y_test1)))


plt.scatter(X_train1[:,0],X_train1[:,1],c=cm[y_train1],s=50,edgecolors='none')
plt.scatter(X_test1[:,0],X_test1[:,1],c='none',s=50,edgecolors=cm[y_test1])



plt.show()

