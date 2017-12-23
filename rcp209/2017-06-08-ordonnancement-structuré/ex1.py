from abc import ABCMeta, abstractmethod
import numpy as np
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Class corresponding to the query
c = 8
# Generate binary labels for class  c
y_trainb = np.zeros(60000)
y_testb = np.zeros(10000)
y_trainb[y_train==c] = 1
y_trainb[y_train!=c] = 0
y_testb[y_test==c] = 1
y_testb[y_test!=c] = 0

tuples = map(lambda t: (t, y_trainb[t]), range(60000))
trier = [v for v,k in tuples if k == 1]

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



class RankingOutput():
  # Rang/index de tout ceux qui sont pertinents/positifs
  ranking = [] 
  # Nombre d'elements positifs
  nbPlus=0
  def __init__(self,ranking,nbPlus):
      self.nbPlus = nbPlus
      self.ranking = ranking

  def labelsfromranking(self):
      labels = [0 for i in range(len(self.ranking))]
      print "self.nbPlus ", self.nbPlus
      print "len(labels) = ", len(labels)
      for i in range(self.nbPlus):
          print "i = ", i
          print "ranking = ", self.ranking[i]
          labels[self.ranking[i]]=1
      return labels
  def __getitem__(self, key):
      return self


import numpy as np
from keras.utils import np_utils
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

class SRanking(SModel):

    def __init__(self, rankingoutput, d, size, learning_rate=0.1, epochs=2):
        SModel.__init__(self,learning_rate, epochs, size)

        # Ground Truth labels
        self.rankingoutput = rankingoutput
        self.d = d

        self.w = np.array([0 for i in range(d)])
        print "w=",self.w.shape, "d=",d, "nb ex=",len(self.rankingoutput.ranking), " nbPlus=",self.rankingoutput.nbPlus

    def predict(self, X):
        # Inference: sorting elts wrt <w,x>
        pred = -1.0*np.matmul(X,self.w)
        # When predicting, number of + unknown, setting it to default value 0
        return RankingOutput(list(np.argsort(pred[:])),0)


    def delta(self, y1, y2):
        # y2 assumed to be ground truth, y1 predicted ranking
        labels = y2.labelsfromranking()
        pred = [0 for i in range(len(y1.ranking)) ]
        for i in range(len(y1.ranking)):
            pred[y1.ranking[i]] = -i

        return 1.0- average_precision_score(labels, pred)

    def yij(self, i, j, d):
        if i == j:
          return 0
        return 1 if d[i] < d[j] else -1

    def psi(self, X, output):
      d = self.rankingoutput.labelsfromranking()
      enum = d.enumerate()
      return sum(map(lambda i: sum(map(lambda j: self.yij(i, j, d) * (X[i] - X[j])), [i for i, d in enum if i == -1]), [i for i, d in enum if i == +1] ))        

    def lai(self, X, Y):
      # Computing <w; x_i> for all examples
      pred = np.matmul(X,self.w)

      # Getting corresponding labels from ranking output Y
      labels = Y.labelsfromranking()

      # Dict structure
      dict = {i : pred[i] for i in range(len(labels))}

      iplus = np.where(np.array(labels[:])==1)[0]
      iminus = np.where(np.array(labels[:])==0)[0]
      nbPlus = len(iplus)
      nbMinus = len(iminus)
      nbTot = nbPlus+nbMinus

      dictplus = {i:dict[i] for i in iplus}
      dictminus = {i:dict[i] for i in iminus}

      # Sorting dict for + examples and - examples
      dictplus_s = sorted(dictplus.iteritems(), key=lambda (k,v): (v,k), reverse=True)
      dictminus_s = sorted(dictminus.iteritems(), key=lambda (k,v): (v,k), reverse=True)

      # Initializing list with + examples
      list_lai = [dictplus_s[i][0] for i in range(len(dictplus_s))]
      # Computing delta2 for each (+/-) pair in Eq 5
      delta1 = 1.0/nbPlus* np.array( [[ ( float(j) / float(j+i) - (j-1.0)/(j+i-1.0)) -
      ( 2.0* (dictplus_s[i-1][1] - dictminus_s[j-1][1])/(nbMinus) ) for j in range(1,nbMinus+1) ] for i in range(1,nbPlus+1)])
      delta2 = delta1
      for i in range(nbPlus-2,-1,-1):
        delta2[i,:] = delta2[i,:] +  delta2[i+1,:]
      # i* = argmax_i delta2 for all - examples
      res = np.argmax(delta2,axis=0)

      # Creating the final list
      for j in range(len(iminus)):
        list_final.insert(res[j]+j, dictminus_s[j][0])

      # Returning it as a RankingOutput class
      return RankingOutput(list_lai,Y.nbPlus)


print "len(trier) = ", len(trier)
print y_trainb.tolist().count(1)
y_train = RankingOutput(ranking = trier, nbPlus = y_trainb.tolist().count(1))
model = SRanking(y_train, d=784, size=60000)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)


tuples = map(lambda t: (t, y_testb[t]), range(10000))
trier = [v for v,k in tuples if k == 1]
ystar = RankingOutput(ranking = trier, nbPlus = y_testb.tolist().count(1))

print "pref test =",(1.0-model.delta(y_predict, y_test))*100.0
