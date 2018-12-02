import util

(x_train, y_train, x_test, y_test) = util.load_data()

# Les réseaux convolutifs manipulent des images multi-dimensionnelles en entrée (tenseurs).
# On reformate les données d’entrée afin que chaque exemple soit de taille 28×28×1.
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

model = Sequential()
# LeNet
# - couche de convolution avec 16 filtres de taille 5×5 suivi d'une non linéarité de type relu
# - couche de max pooling de taille 2×2.
# - couche de convolution avec 32 filtres de taille 5×5, suivie d’une non linéarité de type relu
# - couche de max pooling de taille 2×2.
# - “mettre à plat” les couches convolutives précédentes
# - couche complètement connectée de taille 100, suivie d’une non linéarité de type sigmoïde
# - couche complètement connectée de taille 10, suivie d’une non linéarité de type softmax

# Des couches de convolution, qui transforment un
# tenseur d’entrée de taille n_x×n_y×p => tenseur de sortie nx′×ny′×nH
#  - 16 est le nombre de filtres.
#  - (5, 5) est la taille spatiale de chaque filtre (masque de convolution).
#  - padding=’valid’ correspond ignorer les bords lors du calcul
#    (et donc à diminuer la taille spatiale en sortie de la convolution).
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1), padding='valid'))
# On passe de 28 x 28 => 24 x 24 : perte de pixels
# 2 de perte car kernel size = 5x5 =>
# xxxxx
# xxxxx
# xx***-- => on perd les deux lignes de chaque bord
# xx***--
# xx***--
# couches d’agrégation spatiale (pooling), afin de permettre une invariance aux translations locales.
model.add(MaxPooling2D(pool_size=(2, 2)))
# Max pooling de 2 => division par 2 la taille
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(12, 12, 1), padding='valid'))
# On passe de 12 x 12 => 8 x 8 : perte de pixels
model.add(MaxPooling2D(pool_size=(2, 2)))
# On passe de 8 x 8 => 4 x 4 : perte de pixels
model.add(Flatten())
# 4 * 4 pixels * 32 filtres = 512
model.add(Dense(100, input_dim=512, name='fc-cache'))
model.add(Activation('sigmoid'))
model.add(Dense(10, input_dim=100, name='fc-sortie'))
model.add(Activation('softmax'))

model.summary()

from keras.utils import np_utils

K = 10
# convert class vectors to binary class matrices
# générer des labels au format 0-1 encoding
Y_train = np_utils.to_categorical(y_train, K)
Y_test = np_utils.to_categorical(y_test, K)

from keras.optimizers import SGD

learning_rate = 0.5
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

batch_size = 100
nb_epoch = 50
model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)

scores = model.evaluate(x_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
