#!/usr/bin/python
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np

class SModel:
  _metaclass__ = ABCMeta
  w=[]
  # Batch size
  tbatch = 10
  nbatch=0
  epochs = 10
  # Gradient step
  eta = 0.1
  # Regularization parameter
  llambda = 1e-6

  # Joint feature map Psi(x,y) in R^d
  @abstractmethod
  def psi(self, X, Y):
      pass

  # Loss-Augmented Inference (LAI):  arg max_y [<w, Psi(x,y)> + Delta(y,y*)]
  @abstractmethod
  def lai(self, X, Y):
     pass

  # Inference: arg max_y [<w, Psi(x,y)>]
  @abstractmethod
  def predict(self, X):
      pass

  @abstractmethod
  # Loss between two outputs
  def delta(self, Y1, Y2):
      pass

  def __init__(self, learning_rate=0.1, epochs=2, tbatch=100):
      self.eta=learning_rate
      self.epochs = epochs
      self.tbatch = tbatch


  def fit(self, X, Y):

      self.nbatch = X.shape[0]  / self.tbatch

      for i in range(self.epochs):
          pred = self.predict(X)
          #print "pred = ", pred
          #print "Y = ", Y
          print "epoch ", i, " pref train=",(1.0-self.delta(pred, Y))*100.0, " norm W=",np.sum(self.w*self.w)
          for b in range(self.nbatch):
              # Computing joint feature map psi for groud truth output
              x = X[b*self.tbatch:(b+1)*self.tbatch,:]
              ystar = Y[b*self.tbatch:(b+1)*self.tbatch]
              
              # Computing most violated constraint => LAI
              yhat = self.lai(x, ystar)
              print "yhat = ", yhat
              # Computing joint feature map psi for LAI output
              # print "yhat = ", yhat
              # print "ystart = ", ystar
              gi = self.psi(x, yhat) - self.psi(x, ystar)

              # Computing gradient
              grad = self.llambda*self.w + (gi)

              self.w = self.w - 1.0*self.eta * (grad)
              print "w = " ,self.w


class SXclass(SModel):
  d = 0
  K = 0

  def __init__(self, d, K, learning_rate=0.1, epochs=2, tbatch=100):
    SModel.__init__(self,learning_rate, epochs, tbatch)
    self.d = d
    self.K = K
    self.w = np.zeros((d, K))

  def __str__(self):
    return "SXclass - size w="+str(self.w.shape)+ " eta="+str(self.eta)+" epochs="+ str(self.epochs)+ " tbatch="+str(self.tbatch)


  # retourne dans R^d
  def psi(self, X, Y):
    app = map(lambda t: self._psi(X[t], Y[t]),  range(0, self.tbatch))
    som = reduce(np.add, app, np.zeros((self.d, self.K)))
    return som / self.tbatch

  def _psi(self, X, Y):
    m = np.zeros((self.d, self.K))
    m[:, Y] = X
    print "Y = ", Y
    print "X = ", X
    print m
    return m

  # Loss-Augmented Inference (LAI):  arg max_y [<w, Psi(x,y)> + Delta(y,y*)]
  def lai(self, X, Y):
    app = map(lambda t: self._lai(X[t], Y[t]),  range(0, self.tbatch))
    return np.array(app)
    
  def _lai(self, X, Y):
    v = map(lambda y: (y, np.trace(np.dot(np.transpose(self.w), self._psi(X, y))) + self.delta(y, Y)), range(0, self.K))
    m = max(v, key=lambda t: t[1])
    return m[0]

  # Inference: arg max_y [<w, Psi(x,y)>]
  # Retourne un tableau R^(X.shape[0])
  def predict(self, X):
    app = map(lambda t: self._predict(X[t]),  range(0, X.shape[0]))
    return app

  def _predict(self, X):
    v = map(lambda y: (y, np.trace(np.dot(np.transpose(self.w), self._psi(X, y)))), range(0, self.K) )
    m = max(v, key=lambda t: t[1])
    return m[0]

  # Loss between two outputs
  def delta(self, Y1, Y2):
    return 1 if (Y1 == Y2).all() else 0

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

model = SXclass(d = 784, K = 10)
model.fit(X_train, y_train)


y_predict = model.predict(X_test)
print "pref test =",(1.0-model.delta(y_predict, y_test))*100.0
