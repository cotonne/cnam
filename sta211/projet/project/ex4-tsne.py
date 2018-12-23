import pandas as pd
from sklearn import manifold
from sklearn import preprocessing

from util import convexHulls, best_ellipses, visualization

def codageDisjonctifComplet(X, name):
    values = X[name].values.reshape(-1, 1)
    le = preprocessing.LabelEncoder()
    oneHotEncodedValues = le.fit_transform(values).reshape(-1, 1)
    from sklearn.preprocessing import OneHotEncoder
    onehotencoder = OneHotEncoder()
    features = pd.DataFrame(onehotencoder.fit_transform(oneHotEncodedValues).toarray())
    features.columns = onehotencoder.get_feature_names()
    X = X.drop(columns=[name])
    return X.join(features, lsuffix=name)


filename = 'data_train_ncp_15.csv'
delimiter = ';'

X = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False)
X = X.drop("lvefbin", 1)
X = pd.DataFrame(X, columns=X.columns)
X = codageDisjonctifComplet(X, "centre")
X = codageDisjonctifComplet(X, "country")
X = preprocessing.StandardScaler().fit_transform(X)

labels = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False,
                usecols=["lvefbin"],
                dtype={"lvefbin": 'category'})
# from Pandas Dataframe to numpy dataframe
labels = labels.values.reshape(1, -1)[0]
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)
# Enveloppe convexe des points projetés pour chacune des classes
# ellipse de meilleure approximation des points
# Neighborhood Hit:
#   Pour chaque point, la métrique NH consiste à calculer,
#   pour les k plus proches voisins (k-nn) de ce point,
#   le taux de voisins qui sont de la même classe que le point considéré.
#   La métrique NH est ensuite moyennée sur l’ensemble de la base.

# En quoi les trois métriques ci-dessus sont-elles liées au problème de la séparabilité des classes ?
# Qu’est-ce qui les diffère ?
# enveloppe : tous les points ==> si séparer, tous les points sont séparés
# ellipse de meilleure approximation des points : ellipse qui contient 95% des points ==> si séparer, bonnes séparation des classes en moyennes
# NH : + NH est grand, plus un point est proche de ces voisins => bonne concentration
# Lien entre convex hull/enveloppe convexe & ellipse (convexe hull incluse dans ellipse)?
#

tsne = manifold.TSNE(init='pca', perplexity=50, verbose=2)
x2d = tsne.fit_transform(X)
convex_hulls = convexHulls(x2d, labels)
ellipses = best_ellipses(x2d, labels)
visualization(x2d, labels, convex_hulls, ellipses, "", 1)
