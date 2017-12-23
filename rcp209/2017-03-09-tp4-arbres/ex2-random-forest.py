from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import sys
import matplotlib.pyplot as plt

mult = 201
num_cores = 8

digits = load_digits()
X=digits.data
y=digits.target

def evaluate(n_estimators):
	sys.stdout.write('.')
	sys.stdout.flush()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)
	clf = RandomForestClassifier(n_estimators=n_estimators)
	clf.fit(X_train, y_train)
	Z = clf.predict(X_test)
	return 1 - clf.score(X_test,y_test)

def evaluate_extra(n_estimators):
	sys.stdout.write('.')
	sys.stdout.flush()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)
	clf = ExtraTreesClassifier(n_estimators=n_estimators)
	clf.fit(X_train, y_train)
	Z = clf.predict(X_test)
	return 1 - clf.score(X_test,y_test)

accuracies = np.array(Parallel(n_jobs=num_cores)(delayed(evaluate)(i) for i in range(1, mult)))

avg = np.average(accuracies)
std = np.std(accuracies)

print()
print("Average error = {0}, standard deviation = {1}".format(avg, std))

accuracies_extra = np.array(Parallel(n_jobs=num_cores)(delayed(evaluate_extra)(i) for i in range(1, mult)))

avg = np.average(accuracies_extra)
std = np.std(accuracies_extra)

print()
print("EXTRA Average error = {0}, standard deviation = {1}".format(avg, std))

f = plt.figure()
plt.scatter(range(1, mult), accuracies, c='r')
plt.scatter(range(1, mult), accuracies_extra, c='b')
plt.show()