#!/usr/bin/python
# -*- coding: utf-8 
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import model_from_yaml
from keras.layers import Dense, Activation
from keras.datasets import mnist

nbHidden = 100

def loadModel(savename):
  with open(savename+".yaml", "r") as yaml_file:
    model = model_from_yaml(yaml_file.read())
  print "Yaml Model ",savename,".yaml loaded "
  model.load_weights(savename+".h5")
  print "Weights ",savename,".h5 loaded "
  return model

def saveModel(model, savename):
  # serialize model to YAML
  model_yaml = model.to_yaml()
  with open(savename+".yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
    print "Yaml Model ",savename,".yaml saved to disk"
  # serialize weights to HDF5
  model.save_weights(savename+".h5")
  print "Weights ",savename,".h5 saved to disk"


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

model = loadModel("MLP_100")
model.summary()

model.pop()

# Prevent from being trainable
for layer in model.layers:
	layer.trainable = False

model.add(Dense(nbHidden, activation="sigmoid", name='hidden_1'))
model.add(Dense(2, activation="sigmoid", name='hidden_2'))

model.add(Dense(10,  input_dim=2, name='final',
	kernel_initializer = 'random_normal',
	bias_initializer   = 'random_normal'))
# ajout d’une couche d’activation de type softmax
model.add(Activation('softmax', name='softmax'))

learning_rate = 0.5
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

batch_size = 300
nb_epoch = 10
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1)


saveModel(model, "model1")