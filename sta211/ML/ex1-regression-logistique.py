# coding=utf-8

def load_data():
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # On normalise le vecteur (valeur dans l'intervalle [0,1]
    X_train /= 255
    X_test /= 255
    return (X_train, y_train, X_test, y_test)

# Nombre d'entrées : 784
# Nombre de sorties : 10
# Nombre de poids Wij: 784 * 10
# Nombre de biais: 10
# Nombre de paramètres du modèle : 7850

# On créé un réseau de neurones vide.
from keras.models import Sequential
model = Sequential()

#  On ajoute des couches avec la fonction add.
from keras.layers import Dense, Activation
#  - couche de projection linéaire (couche complètement connectée) de taille 10
model.add(Dense(10,  input_dim=784, name='fc1'))
#  - couche d’activation de type softmax (interprétation: proba d'appartenir à une classe)
model.add(Activation('softmax'))

# visualisation de l'architecture
model.summary()

(X_train, Y_train, X_test, Y_test) = load_data()

from keras.utils import np_utils
K=10
# convert class vectors to binary class matrices
# générer des labels au format 0-1 encoding
Y_train = np_utils.to_categorical(Y_train, K)
Y_test = np_utils.to_categorical(Y_test, K)

# our mesurer l’erreur de prédiction, on utilisera une fonction de
# coût de type entropie croisée (“cross-entropy”)
# La fonction de coût de l’Eq. (3) est-elle convexe par rapports aux paramètres W, b du modèle ? Non
# Avec un pas de gradient bien choisi, peut-on assurer la convergence vers le minimum global de la solution ? Non

# Afin d’optimiser les paramètres W et b pour minimiser la fonction de coût
# pour notre modèle de régression logistique, nous allons utiliser l’algorithme
# de rétro-propagation de l’erreur du gradient.
# Avec Keras, la rétro-propagation de l’erreur est implémentée nativement.
# On va compiler le modèle en lui passant un loss (ici l’ entropie croisée),
# une méthode d’optimisation (ici une descente de gradient stochastique, stochatic gradient descent, sgd),
# et une métrique d’évaluation (ici le taux de bonne prédiction des catégories, accuracy) :
from keras.optimizers import SGD
learning_rate = 0.5
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

batch_size = 100
nb_epoch = 10
model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1)

scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))