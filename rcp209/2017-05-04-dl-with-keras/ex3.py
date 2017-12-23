#!/usr/bin/python
# -*- coding: utf-8 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.callbacks import TensorBoard
from keras.models import model_from_yaml
def saveModel(model, savename):
  # serialize model to YAML
  model_yaml = model.to_yaml()
  with open(savename+".yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
    print "Yaml Model ",savename,".yaml saved to disk"
  # serialize weights to HDF5
  model.save_weights(savename+".h5")
  print "Weights ",savename,".h5 saved to disk"


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


# Transforme les images 32 x 32 ==> 28 x 28 (dû )
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

model = Sequential()

# ajout d’une couche de projection linéaire (couche complètement connectée) de taille 10
model.add(Conv2D(32,kernel_size=(5, 5),activation='sigmoid',input_shape=(28, 28, 1),padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32,kernel_size=(5, 5),activation='sigmoid',input_shape=(14, 14, 1),padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(84,  input_dim=120, name='fc1',
	kernel_initializer = 'random_normal',
	bias_initializer   = 'random_normal'))
model.add(Activation('sigmoid'))
model.add(Dense(10,  input_dim=84, name='fc2',
	kernel_initializer = 'random_normal',
	bias_initializer   = 'random_normal'))
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

saveModel(model, "output")

scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

