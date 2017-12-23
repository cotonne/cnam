#!/usr/bin/python
# -*- coding: utf-8 

from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

def forward(batch, Wh, bh, Wy, by):
  xt = batch
  mh = np.dot(xt, Wh)
  ah = mh + bh
  yh = 1 / (1 + np.exp(-ah))

  my = np.dot(yh, Wy)
  ay = my + by
  yy = np.apply_along_axis(softmax, 1, ay)

  return (yy, yh)
    
def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    
    Rows are scores for each class. 
    Columns are predictions (samples).
    """
    scoreMatExp = np.exp(x)
    return scoreMatExp / scoreMatExp.sum(0)

def accuracy(Wh, bh, Wy, by, images, labels):
    (pred, hidden) = forward(images, Wh,bh, Wy, by )
    return np.where( pred.argmax(axis=1) != labels.argmax(axis=1) , 0.,1.).mean()*100.0

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


K = 10
H = 100
# On génère des labels au format 0-1 encoding - équation (2).
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, K)
Y_test = np_utils.to_categorical(y_test, K)

# Taille
N = X_train.shape[0]
d = X_train.shape[1]
Wy = np.zeros((H,K))
by = np.zeros((1,K))
gradWy = np.zeros((H,K))
gradby = np.zeros((1,K))

Wh = np.zeros((d,H))
bh = np.zeros((1,H))
gradWh = np.zeros((d,H))
gradbh = np.zeros((1,H))


numEp = 40 # Number of epochs for gradient descent
eta = 0.1 # Learning rate
batch_size = 100
nb_batches = int(float(N) / batch_size)

np.set_printoptions(threshold=np.nan)

for epoch in range(numEp):
  print epoch
  print accuracy(Wh, bh, Wy, by, X_test, Y_test)
  for  i in range(nb_batches):
    (batch, expectedOutputs) = (X_train[i*batch_size:(i+1)*batch_size], Y_train[i*batch_size:(i+1)*batch_size])

    (outputs, outputsH) = forward(batch, Wh, bh, Wy, by)
    deltay = outputs - expectedOutputs
    gradWy = (1/float(batch_size)) * np.dot(np.transpose(outputsH), deltay)
    gradby = np.mean(deltay, axis=0)
    Wy = Wy - eta * gradWy
    by = by - eta * gradby

    deltah = np.dot(deltay, np.transpose(Wy)) * outputsH * (1 - outputsH)
    gradWh = (1/float(batch_size)) * np.dot(np.transpose(batch), deltah)
    gradbh = np.mean(deltah, axis=0)
    Wh = Wh - eta * gradWh
    bh = bh - eta * gradbh
    #if epoch % 100 == 0:
    #  print accuracy(Wh, bh, Wy, by, X_test, Y_test)      

print accuracy(Wh, bh, Wy, by, X_test, Y_test)