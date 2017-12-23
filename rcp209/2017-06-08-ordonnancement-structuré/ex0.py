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

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(1,  input_dim=784, name='fc1'))
model.add(Activation('sigmoid'))
model.summary()

from keras.optimizers import SGD

learning_rate = 1.0
sgd = SGD(learning_rate)
model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['binary_accuracy'])

nbex=1000
batch_size = 100
nb_epoch = 20
model.fit(X_train[0:nbex,:], y_trainb[0:nbex],batch_size=batch_size, epochs=nb_epoch,verbose=1)

scorestrain = model.evaluate(X_train[0:nbex,:], y_trainb[0:nbex], verbose=0)
scorestest = model.evaluate(X_test, y_testb, verbose=0)
print("perfs train - %s: %.2f%%" % (model.metrics_names[1], scorestrain[1]*100))
print("perfrs test - %s: %.2f%%" % (model.metrics_names[1], scorestest[1]*100))

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Computing prediction for the training set
predtrain = model.predict(X_train[0:nbex,:])
# Computing precision recall curve
precision, recall, _ = precision_recall_curve(y_trainb[0:nbex], predtrain)
# Computing Average Precision
APtrain = average_precision_score(y_trainb[0:nbex], predtrain)
print "Class ",c," - Average Precision  TRAIN=", APtrain*100.0

plt.clf()
plt.plot(recall, precision, lw=2, color='navy',label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall TRAIN for class '+str(c)+' : AUC={0:0.2f}'.format(APtrain*100.0))
plt.legend(loc="lower left")



plt.figure()
# Computing prediction for the training set
predtest = model.predict(X_test[0:nbex,:])
# Computing precision recall curve
precision, recall, _ = precision_recall_curve(y_testb[0:nbex], predtest)
# Computing Average Precision
APtest = average_precision_score(y_testb[0:nbex], predtest)
print "Class ",c," - Average Precision TEST=", APtest*100.0

plt.clf()
plt.plot(recall, precision, lw=2, color='navy',label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall TEST for class '+str(c)+' : AUC={0:0.2f}'.format(APtest*100.0))

plt.show()