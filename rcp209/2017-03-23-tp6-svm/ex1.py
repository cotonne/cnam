
#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from matplotlib import cm as colormap
from sklearn import svm, datasets
from  sklearn.datasets import load_svmlight_file
from sklearn.decomposition import TruncatedSVD


data = load_svmlight_file("./tpsvm/libsvm-3.22/databases/56.scaled")

X = data[0]
y = data[1]

pca = TruncatedSVD(n_components=50)
pca.fit_transform(X, y)
ratio = np.flip(np.sort(pca.explained_variance_ratio_), 0)
cumulated_ratio = ratio.cumsum()

fig, ax1 = plt.subplots()
ax1.plot(np.arange(0, 50, 1), ratio, 'b')
ax1.set_ylabel("ratio")
ax2 = ax1.twinx()
ax2.plot(np.arange(0, 50, 1), cumulated_ratio, 'r')
ax2.set_ylabel("Cumulated ratio")
fig.tight_layout()
plt.show()