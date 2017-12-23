#!/usr/bin/python
# -*- coding: utf-8 

from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

def forward(batch, W, b):
	xt = batch
	m = np.dot(xt, W)
	a = m + b
	return np.apply_along_axis(softmax, 1, a)
		
def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    
    Rows are scores for each class. 
    Columns are predictions (samples).
    """
    scoreMatExp = np.exp(x)
    return scoreMatExp / scoreMatExp.sum(0)

def accuracy(W, b, images, labels):
    pred = forward(images, W,b )
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


K=10
# On génère des labels au format 0-1 encoding - équation (2).
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, K)
Y_test = np_utils.to_categorical(y_test, K)

# Taille
N = X_train.shape[0]
d = X_train.shape[1]
W = np.zeros((d,K))
b = np.zeros((1,K))
numEp = 20 # Number of epochs for gradient descent
eta = 0.1 # Learning rate
batch_size = 100
nb_batches = int(float(N) / batch_size)
gradW = np.zeros((d,K))
gradb = np.zeros((1,K))

for epoch in range(numEp):
  print accuracy(W, b, X_train, Y_train)
  for  i in range(nb_batches):
    (batch, expectedOutputs) = (X_train[i*batch_size:(i+1)*batch_size], Y_train[i*batch_size:(i+1)*batch_size])
    outputs = forward(batch, W, b)
    delta = outputs - expectedOutputs
    gradW = (1/float(batch_size)) * np.dot(np.transpose(batch), delta)
    gradb = np.mean(delta, axis=0)
    W = W - eta * gradW
    b = b - eta * gradb

print accuracy(W, b, X_test, Y_test)