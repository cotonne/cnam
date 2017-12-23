#!/usr/bin/python
# -*- coding: utf-8 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import SGD

from keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir="_mnist", write_graph=False, write_images=True)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')  

model = Sequential()

# ajout d’une couche de projection linéaire (couche complètement connectée) de taille 10
model.add(Dense(10,  input_dim=784, name='fc1'))
# ajout d’une couche d’activation de type softmax
model.add(Activation('softmax'))

print model.summary()

learning_rate = 0.5
# méthode d’optimisation : descente de gradient stochastique, stochatic gradient descent, sgd
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

batch_size = 300
nb_epoch = 10
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
# Apprentissage
#model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1)
model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1, callbacks=[tensorboard])


scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

