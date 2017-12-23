from sklearn import tree
clf = tree.DecisionTreeClassifier()
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = clf.fit(X, Y)
print(clf.predict([[2., 2.]]))
print(clf.predict_proba([[2., 2.]]))