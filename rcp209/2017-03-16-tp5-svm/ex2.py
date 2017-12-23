# encoding: utf8
from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
digits = load_digits()


# def parse(line):
#     number = int(line[:0])
#     rest = line[1:].split(" ")

# with open("tpsvm/libsvm-3.22/databases/mnist") as f:
#     lines = f.readlines()

# all_data = [parse(line) for line in lines]

# X = all_data[:0]
# Y = all_data[0:]

digits = load_digits()

X=digits.data
y=digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Pour afficher la surface de décision on va discrétiser l'espace avec un pas h
h = .02
C = 1 # paramètre de régularisation
svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
print "Result  SVC " 
print svc.score(X_test, y_test)
lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)
print "Result Lin SVC " 
print lin_svc.score(X_test, y_test)
