# coding=utf-8

# Modèle multiple : 1 / centre
#
# Pour le jeu d'apprentissage:
# - 3 modèles : 1 pour le centre 7, 1 pour le centre 8 et 1 pour le reste
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

Y_train = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False,
                      usecols=["lvefbin"],
                      dtype={"lvefbin": 'S4'})

models = [build(X_train[X_train['cluster'] == i].drop("cluster", 1), Y_train[X_train['cluster'] == i]) for i in
          np.unique(X_train['cluster'])]


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