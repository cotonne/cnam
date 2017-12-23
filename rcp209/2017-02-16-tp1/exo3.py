import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
reg = linear_model.LinearRegression()


# definir matrices de rotation et de dilatation
rot = np.array([[0.94, 0.34], [-0.34, 0.94]])
et2 = 3
sca = np.array([[10, 0], [0, et2]])
# generer donnees classe 1
np.random.seed(60)
datar = (np.random.randn(60,2)).dot(sca).dot(rot)


X_train1, X_test1, y_train1, y_test1 = train_test_split(datar[:,0], datar[:,1], test_size=0.33)

# evaluation et affichage sur split1
reg.fit(X_train1.reshape(-1,1), y_train1)
# attention, pas erreur mais coeff. determination !
reg.score(X_train1.reshape(-1,1), y_train1)

Y_predict_train1 = reg.predict(X_train1.reshape(-1, 1))
print("Erreur quadratique apprentissage = " + str(np.sum([(y - yp) ** 2 for (y, yp) in zip(y_train1, Y_predict_train1)])/len(X_train1)))
Y_predict_test1 = reg.predict(X_test1.reshape(-1, 1))
print("Erreur quadratique test = " + str(np.sum([(y - yp) ** 2 for (y, yp) in zip(y_test1, Y_predict_test1)])/len(X_test1)))

reg.score(X_test1.reshape(-1,1), y_test1)

plt.scatter(X_train1,y_train1,s=50,edgecolors='none')
plt.scatter(X_test1,y_test1,c='none',s=50,edgecolors='blue')
nx = 100
x_min, x_max = plt.xlim()
xx = np.linspace(x_min, x_max, nx)

for size in np.arange(0.1, 0.9, 0.1):
  X_train1, X_test1, y_train1, y_test1 = train_test_split(datar[:,0], datar[:,1], test_size=0.33)
  reg.fit(X_train1.reshape(-1,1), y_train1)
  plt.plot(xx,reg.predict(xx.reshape(-1,1)),color='black')

plt.show()
