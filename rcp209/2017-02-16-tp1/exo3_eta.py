import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
reg = linear_model.LinearRegression()

def calc(et2):
    # définir matrices de rotation et de dilatation
    rot = np.array([[0.94, 0.34], [-0.34, 0.94]])
    sca = np.array([[10, 0], [0, et2]])
    # générer données classe 1
    np.random.seed(60)
    datar = (np.random.randn(60,2)).dot(sca).dot(rot)
    
    
    X_train1, X_test1, y_train1, y_test1 = train_test_split(datar[:,0], datar[:,1], test_size=0.33)
    
    # évaluation et affichage sur split1
    reg.fit(X_train1.reshape(-1,1), y_train1)
    # attention, pas erreur mais coeff. détermination !
    return reg.score(X_test1.reshape(-1,1), y_test1)

et2s = [1, 2, 4, 6]
coeffs = [calc(x) for x in et2s]
plt.plot(et2s, coeffs,color='black')
    
plt.show()
