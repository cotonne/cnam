# coding=utf-8

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

filename = 'data_train_ncp_15.csv'
filename_test = "data_test_ncp_15.csv"
csv_res = "python_mlp.csv"
delimiter = ';'
data = []


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


X = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False)
X = X.drop("lvefbin", 1)
X = pd.DataFrame(X, columns=X.columns)
X = codageDisjonctifComplet(X, "centre")
X = codageDisjonctifComplet(X, "country")
X = preprocessing.StandardScaler().fit_transform(X)

y = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False,
                usecols=["lvefbin"],
                dtype={"lvefbin": 'category'})

X_train, X_test, Y_train, Y_test = train_test_split(X, y.values.ravel(), test_size=0.20, random_state=42)

from keras.models import Sequential

model = Sequential()

from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2

model.add(Dense(100, input_dim=X.shape[1], name='fc-cache1', kernel_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1, input_dim=100, name='fc-sortie'))
#  - couche d’activation de type softmax
model.add(Activation('sigmoid'))

# visualisation de l'architecture
model.summary()

K = 2
# convert class vectors to binary class matrices
# générer des labels au format 0-1 encoding
labels, Y_train = np.unique(Y_train, return_inverse=True)

print(Y_train)

from keras.optimizers import SGD

learning_rate = 0.5
sgd = SGD(learning_rate)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'binary_crossentropy'])

batch_size = 100
nb_epoch = 1000
history = model.fit(X_train, Y_train, validation_split=0.2, batch_size=batch_size, epochs=nb_epoch, verbose=1)

#
# PREDICTION
#
pred_test = model.predict(X_test, verbose=1)
col = pred_test  # [:, 1]
col_bis = (col >= 0.5).astype(int)
Y_pred = pd.DataFrame(labels[col_bis])
print(confusion_matrix(Y_test, Y_pred))
# df.to_csv("python_mlp.csv")

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

import util

util.saveModel(model, "model-ex2")

# La fonction de coût de l’Eq. (3) est-elle convexe par rapports aux paramètres W, b du modèle ? Non
# Avec un pas de gradient bien choisi, peut-on assurer la convergence vers le minimum global de la solution ? Non

# Façon dont les paramètres du modèles sont initialisés dans les différentes couches
# model.add(Dense(64,
#                 kernel_initializer='random_uniform',
#                 bias_initializer='zeros'))
