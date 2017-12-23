#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numpy import genfromtxt

import csv
import collections
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2 as chi2_s
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from matplotlib import cm as colormap
import pydotplus
from mpl_toolkits.mplot3d import Axes3D
from keras.callbacks import TensorBoard


sns.set(style="whitegrid", color_codes=True)


# Reading CSV
filename = 'german.data.csv'
delimiter = ' '
data = []

continuous_values = pd.read_csv(filename, delimiter=delimiter, 
    names=['Duration in month'],
    usecols=[1],
    dtype='float')
discrete_values = pd.read_csv(filename,delimiter=delimiter,
    names=['Status of checking account', 'Credit history', 
         'Savings account', 'Present employment since', 
         'Property', 'Foreigner', 'Credit'],
    usecols=[0,2,5,6,11,19,20],
    dtype='S4')

continuous_values = pd.read_csv(filename, delimiter=delimiter, 
    names=['Duration in month', 'Credit amount', 'Installement rate', 'Present residence since', 'Age', 'Number of existing credits', 'Nb of liable people'],
    usecols=[1,4, 7, 10, 12, 15, 17],
    dtype='float')
discrete_values = pd.read_csv(filename,delimiter=delimiter,
    names=['Status of checking account', 'Credit history', 
         'Purpose', 'Savings account', 'Present employment since', 
         'Personal status and sex', 'Guarantors', 'Property',
         'Other installment plans', 'Housing', 
         'Job', 'Telephone', 'Foreigner', 'Credit'],
    usecols=[0,2,3,5,6,8,9,11,13,14,16,18,19,20],
    dtype='S4')


credit_values = discrete_values['Credit']
discrete_values = discrete_values.drop(labels='Credit', axis=1)

# Reprojection
dummies = pd.get_dummies(discrete_values.ix[:,:'Foreigner'], columns=discrete_values.columns.values, drop_first = True)

data = pd.concat([continuous_values, dummies], axis=1)
minmax = MinMaxScaler()
X = minmax.fit_transform(data)
y = credit_values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

y_train = np_utils.to_categorical(y_train, 3)
y_train = y_train[:,[1, 2]]
y_test = np_utils.to_categorical(y_test, 3)
y_test = y_test[:,[1, 2]]
tensorboard = TensorBoard(log_dir="_tensorboard", write_graph=True, write_images=True)

print data.shape[1]
model = Sequential()
model.add(Dense(50,  input_dim=data.shape[1], name='hidden',
  kernel_initializer = 'random_normal',
  bias_initializer   = 'random_normal'))
model.add(Activation('relu'))
model.add(Dense(2,  input_dim=50, name='fc1',
  kernel_initializer = 'random_normal',
  bias_initializer   = 'random_normal'))
model.add(Activation('softmax'))


print model.summary()

learning_rate = 0.1
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

batch_size = 200
nb_epoch = 1000

model.fit(X_train, y_train,batch_size=batch_size, 
  epochs=nb_epoch, callbacks=[tensorboard])


scores = model.evaluate(X_train, y_train)
print
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
scores = model.evaluate(X_test, y_test)
print
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))