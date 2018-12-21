# coding=utf-8

## hyperparametres
from sklearn.metrics import accuracy_score

batch_size = 100
nb_epoch = 1000
learning_rate = 0.5


def build(x, y):
    tuned_parameters = {
        'kernel': ['rbf'],  # 'linear', 'poly', 'rbf', 'poly'
        'degree': range(2, 5),
        'gamma': np.array(list(map(lambda l: 10 ** (-l), range(-4, 4))))
    }

    from sklearn.model_selection import GridSearchCV
    clf = GridSearchCV(
        svm.OneClassSVM(),
        tuned_parameters,
        # cv=5,
        n_jobs=4,
        # verbose=2,
        # scoring=scoring
        scoring="accuracy"
    )

    clf.fit(x, y)

    return clf.best_estimator_


import numpy as np
import pandas as pd
from sklearn import svm

filename = 'clean_data_train_after_som.csv'
filename_test = "clean_data_test_after_som.csv"
csv_res = "python_svm.csv"
delimiter = ';'
data = []

X = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False)
X = X.drop("lvefbin", 1)
# X_train = preprocessing.StandardScaler().fit_transform(X)
X_train = pd.DataFrame(X, columns=X.columns)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=6, n_init=10, init='random').fit(X_train)
X_train['cluster'] = kmeans.labels_

Y_train = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False,
                      usecols=["lvefbin"],
                      dtype={"lvefbin": 'S4'})
labels, Y_train = np.unique(Y_train, return_inverse=True)

print("Taille cluster")
print(np.unique(kmeans.labels_, return_counts=True))

models = [build(X_train[X_train['cluster'] == i].drop("cluster", 1), Y_train[X_train['cluster'] == i]) for i in
          np.unique(kmeans.labels_)]


def predict(i):
    y_predict = models[i].predict(X_train[X_train['cluster'] == i].drop("cluster", 1))
    col = np.array(y_predict)
    col_bis = (col >= 0.5).astype(int)
    y_true = Y_train[X_train['cluster'] == i]
    from sklearn.metrics import accuracy_score
    return accuracy_score(col_bis, y_true)

print("Apprentissage / Cluster")
print([predict(i) for i in np.unique(kmeans.labels_)])

#
# PREDICTION
#
X_clean = pd.read_csv(filename_test, header=0, sep=delimiter, error_bad_lines=False)
# X_test = preprocessing.StandardScaler().fit_transform(X_clean)
X_test = X_clean  # pd.DataFrame(X_clean, columns=X_clean.columns)

predict_kmeans = kmeans.predict(X_test)
X_test['cluster'] = predict_kmeans

col = [models[int(x['cluster'])].predict(x.drop("cluster").to_frame().transpose()) for index, x in
       X_test.iterrows()]
col = np.array(col)
col_bis = (col >= 0.5).astype(int)
df = pd.DataFrame(labels[col_bis])
df.to_csv(csv_res, index=False, encoding='utf-8')

# util.saveModel(model, "model-ex2")

# La fonction de coût de l’Eq. (3) est-elle convexe par rapports aux paramètres W, b du modèle ? Non
# Avec un pas de gradient bien choisi, peut-on assurer la convergence vers le minimum global de la solution ? Non

# Façon dont les paramètres du modèles sont initialisés dans les différentes couches
# model.add(Dense(64,
#                 kernel_initializer='random_uniform',
#                 bias_initializer='zeros'))
