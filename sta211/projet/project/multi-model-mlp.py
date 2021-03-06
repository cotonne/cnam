# coding=utf-8

# Modèle multiple : 1 / cluster
#
# Pour le jeu d'apprentissage:
# - On détermine des clusters via K-Means
# - Pour chaque cluster, on calcule un modèle de type Réseau de Neurone (MLP)
#   * Une couche cachée de 64 neurones, activation ReLU, régularization et DropOut pour réduire le sur-apprentissage
#   * Une deuxième couche cachée avec les mêmes hyper-paramètres
#   * Une couche de sortie, avec activation sigmoid
#
# Pour le jeu de test
# - On détermine le cluster pour chaque valeur
# - En fonction du cluster, on sélectionne le modèle et on prédit la valeur

def codageDisjonctifComplet(X, name):
    values = X[name].values.reshape(-1, 1)
    le = preprocessing.LabelEncoder()
    oneHotEncodedValues = le.fit_transform(values).reshape(-1, 1)
    from sklearn.preprocessing import OneHotEncoder
    onehotencoder = OneHotEncoder()
    features = pd.DataFrame(onehotencoder.fit_transform(oneHotEncodedValues).toarray())
    features.columns = onehotencoder.get_feature_names()
    X = X.drop(columns=[name])
    return X.join(features, lsuffix=name)


def print_history(history):
    import matplotlib.pyplot as plt
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'loss'], loc='upper left')
    plt.show()


## hyperparametres
batch_size = 100
nb_epoch = 1000
learning_rate = 0.5

def build(x, y):
    #  On créé un réseau de neurones vide.
    from keras.models import Sequential

    model = Sequential()

    #  On ajoute des couches avec la fonction add.
    from keras.layers import Dense, Activation, Dropout
    from keras.regularizers import l2

    # https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
    # L1 regularization, where the cost added is proportional
    # to the absolute value of the weights coefficients (i.e. to what is
    # called the "L1 norm" of the weights).
    #
    # L2 regularization, where the cost added is proportional to the square
    # of the value of the weights coefficients (i.e. to what is called the
    # "L2 norm" of the weights). L2 regularization is also called weight decay
    # in the context of neural networks. Don't let the different name confuse you:
    # weight decay is mathematically the exact same as L2 regularization.

    #  - couche de projection linéaire (couche complètement connectée) de taille 10
    # keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
    model.add(Dense(64, input_dim=x.shape[1], name='fc-cache1', kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, input_dim=64, name='fc-cache2', kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, input_dim=64, name='fc-sortie'))
    #  - couche d’activation de type softmax
    model.add(Activation('sigmoid'))

    # visualisation de l'architecture
    model.summary()

    from keras.optimizers import SGD

    sgd = SGD(learning_rate)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'binary_crossentropy'])

    history = model.fit(x, y, validation_split=0.2, batch_size=batch_size, epochs=nb_epoch, verbose=0)

    return model


import numpy as np
import pandas as pd
from sklearn import preprocessing

filename = 'clean_data_train_after_som.csv'
filename_test = "clean_data_test_after_som.csv"
csv_res = "python_mlp.csv"
delimiter = ';'
data = []

# TODO: create a LabelEncoder object and fit it to each feature in X
X = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False)
X = X.drop("lvefbin", 1)
# X_train = preprocessing.StandardScaler().fit_transform(X)
X_train = pd.DataFrame(X, columns=X.columns)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=6, n_init=10, init='k-means++').fit(X_train)

X_train['cluster'] = kmeans.labels_

Y_train = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False,
                      usecols=["lvefbin"],
                      dtype={"lvefbin": 'S4'})
labels, Y_train = np.unique(Y_train, return_inverse=True)

print("Taille cluster")
group, size_clusters = np.unique(kmeans.labels_, return_counts=True)


xx = np.concatenate((kmeans.labels_.reshape(1, -1), Y_train.reshape(1, -1)), axis=0).transpose()
xx = xx[xx[:, 1] == 1]
group, count_1 = np.unique(xx, return_counts=True, axis=0)

print(count_1/size_clusters)

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

col = [models[int(x['cluster'])].predict(x.drop("cluster").to_frame().transpose())[0][0] for index, x in
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
