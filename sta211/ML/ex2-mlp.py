# coding=utf-8

import util

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
model.add(Dense(100,  input_dim=784, name='fc-cache'))
model.add(Activation('sigmoid'))
model.add(Dense(10,  input_dim=100, name='fc-sortie'))
#  - couche d’activation de type softmax
model.add(Activation('softmax'))

# visualisation de l'architecture
model.summary()

(X_train, Y_train, X_test, Y_test) = util.load_data()

from keras.utils import np_utils
K=10
# convert class vectors to binary class matrices
# générer des labels au format 0-1 encoding
Y_train = np_utils.to_categorical(Y_train, K)
Y_test = np_utils.to_categorical(Y_test, K)

from keras.optimizers import SGD
learning_rate = 0.5
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

batch_size = 100
# nb_epoch = 50
nb_epoch = 5
model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1)

scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

util.saveModel(model, "model-ex2")

# La fonction de coût de l’Eq. (3) est-elle convexe par rapports aux paramètres W, b du modèle ? Non
# Avec un pas de gradient bien choisi, peut-on assurer la convergence vers le minimum global de la solution ? Non

# Façon dont les paramètres du modèles sont initialisés dans les différentes couches
# model.add(Dense(64,
#                 kernel_initializer='random_uniform',
#                 bias_initializer='zeros'))