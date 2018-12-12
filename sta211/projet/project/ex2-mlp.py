# coding=utf-8

from sklearn import preprocessing
import util
import numpy as np
import pandas as pd
# Nombre d'entrées : 784
# Nombre de neurones couches cachées: 100
# Nombre de poids Wij - couche 1: 784 * 100
# Nombre de biais: 100
# Nombre de sorties : 10
# Nombre de poids Wij - couche 1: 100 * 10
# Nombre de biais: 10
# Nombre de paramètres du modèle : 78500 + 1010 = 79510 paramètres

# On créé un réseau de neurones vide.
from keras.models import Sequential
model = Sequential()

#  On ajoute des couches avec la fonction add.
from keras.layers import Dense, Activation
#  - couche de projection linéaire (couche complètement connectée) de taille 10
model.add(Dense(100,  input_dim=14, name='fc-cache'))
model.add(Activation('relu'))
model.add(Dense(2,  input_dim=100, name='fc-sortie'))
#  - couche d’activation de type softmax
model.add(Activation('softmax'))

# visualisation de l'architecture
model.summary()

filename = 'data_train.csv'
delimiter = ';'
data = []


X = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False,
                names=["id", "centre", "country", "gender", "bmi", "age", "egfr", "sbp", "dbp", "hr",
                       "lvef", "lvefbin", "copd", "hypertension", "previoushf", "afib", "cad"],
                usecols=["centre", "country", "gender", "bmi", "age", "egfr", "sbp", "dbp", "hr",
                         "copd", "hypertension", "previoushf", "afib", "cad"],
                dtype={
                    "centre": 'S4',
                    "country": 'S4', "gender": 'S4', "bmi": 'float', "age": 'float', "egfr": 'float',
                    "sbp": 'float', "dbp": 'float', "hr": 'float', "copd": 'S4',
                    "hypertension": 'S4', "previoushf": 'S4', "afib": 'S4', "cad": 'S4'})

Y_train = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False,
                names=["id", "centre", "country", "gender", "bmi", "age", "egfr", "sbp", "dbp", "hr",
                       "lvef", "lvefbin", "copd", "hypertension", "previoushf", "afib", "cad"],
                usecols=["lvefbin"],
                dtype={"lvefbin": 'S4'})

X_test = pd.read_csv("data_test.csv", header=0, sep=";",
                        names=["id", "centre", "country", "gender", "bmi", "age", "egfr", "sbp", "dbp", "hr",
                               "copd", "hypertension", "previoushf", "afib", "cad"],
                        usecols=["centre", "country", "gender", "bmi", "age", "egfr", "sbp", "dbp", "hr",
                                 "copd", "hypertension", "previoushf", "afib", "cad"],
                        dtype={
                            "centre": 'S4',
                            "country": 'S4', "gender": 'S4', "bmi": 'float', "age": 'float', "egfr": 'float',
                            "sbp": 'float', "dbp": 'float', "hr": 'float', "copd": 'S4',
                            "hypertension": 'S4', "previoushf": 'S4', "afib": 'S4', "cad": 'S4'})

X_train = preprocessing.StandardScaler().fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values.ravel(), test_size=0.20, random_state=42)
# (X_train, Y_train, X_test, Y_test) = util.load_data()

from keras.utils import np_utils
K=2
# convert class vectors to binary class matrices
# générer des labels au format 0-1 encoding
labels, Y_train = np.unique(Y_train, return_inverse=True)

print labels
print Y_train
Y_train = np_utils.to_categorical(Y_train, K)

from keras.optimizers import SGD
learning_rate = 0.5
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

batch_size = 100
nb_epoch = 500
model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1)

X_test = preprocessing.StandardScaler().fit_transform(X_test)

pred_test = model.predict(X_test, verbose=1)

col = pred_test[:,1]
col_bis = (col >= 0.5).astype(int)
df = pd.DataFrame(labels[col_bis])
df.to_csv("python_mlp.csv")

util.saveModel(model, "model-ex2")

# La fonction de coût de l’Eq. (3) est-elle convexe par rapports aux paramètres W, b du modèle ? Non
# Avec un pas de gradient bien choisi, peut-on assurer la convergence vers le minimum global de la solution ? Non

# Façon dont les paramètres du modèles sont initialisés dans les différentes couches
# model.add(Dense(64,
#                 kernel_initializer='random_uniform',
#                 bias_initializer='zeros'))
