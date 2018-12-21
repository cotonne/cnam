import numpy as np
import pandas as pd
from sklearn import preprocessing

filename = 'clean_data_train.csv'
delimiter = ';'
data = []

# TODO: create a LabelEncoder object and fit it to each feature in X
X = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False)
X = X.drop("lvefbin", 1)
# X_train = preprocessing.StandardScaler().fit_transform(X)
X_train = pd.DataFrame(X, columns=X.columns)

from sklearn.cluster import KMeans

for i in range(2, 15):
    kmeans = KMeans(n_clusters=i, n_init=10, init='k-means++').fit(X_train)

    X_train['cluster'] = kmeans.labels_

    Y_train = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False,
                          usecols=["lvefbin"],
                          dtype={"lvefbin": 'S4'})
    labels, Y_train = np.unique(Y_train, return_inverse=True)

    print("Taille cluster : {} ".format(i))
    group, size_clusters = np.unique(kmeans.labels_, return_counts=True)


    xx = np.concatenate((kmeans.labels_.reshape(1, -1), Y_train.reshape(1, -1)), axis=0).transpose()
    xx = xx[xx[:, 1] == 1]
    group, count_1 = np.unique(xx, return_counts=True, axis=0)

    print(count_1)
    print(size_clusters)
    print(np.sort(count_1/size_clusters))