import numpy
import pandas as pd

from keras.models import model_from_yaml
from sklearn import preprocessing


def codageDisjonctifComplet(X, name):
    values = X[name].values.reshape(-1, 1)
    le = preprocessing.LabelEncoder()
    oneHotEncodedValues = le.fit_transform(values).reshape(-1, 1)
    from sklearn.preprocessing import OneHotEncoder
    onehotencoder = OneHotEncoder(categories='auto')
    features = pd.DataFrame(onehotencoder.fit_transform(oneHotEncodedValues).toarray())
    features.columns = onehotencoder.get_feature_names()
    X = X.drop(columns=[name])
    return X.join(features, lsuffix=name)


def saveModel(model, savename):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(savename + ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
        print("Yaml Model ", savename, ".yaml saved to disk")
    # serialize weights to HDF5
    model.save_weights(savename + ".h5")
    print("Weights ", savename, ".h5 saved to disk")


from keras.models import model_from_yaml


def loadModel(savename):
    with open(savename + ".yaml", "r") as yaml_file:
        model = model_from_yaml(yaml_file.read())
    print("Yaml Model ", savename, ".yaml loaded")
    model.load_weights(savename + ".h5")
    print("Weights ", savename, ".h5 loaded ")
    return model


def load_data():
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    return (X_train, y_train, X_test, y_test)


def convexHulls(points, labels):
    from scipy.spatial.qhull import ConvexHull
    # computing convex hulls for a set of points with asscoiated labels
    convex_hulls = []
    for i in numpy.unique(labels):
        convex_hulls.append(ConvexHull(points[labels == i, :]))
    return convex_hulls


def best_ellipses(points, labels):
    from sklearn.mixture import GaussianMixture
    # computing best fiiting ellipse for a set of points with asscoiated labels
    gaussians = []
    for i in numpy.unique(labels):
        gaussians.append(GaussianMixture(n_components=1, covariance_type='full').fit(points[labels == i, :]))
    return gaussians


from sklearn.neighbors import NearestNeighbors


def neighboring_hit(points, labels):
    k = 6
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)

    txs = 0.0
    txsc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    nppts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for i in range(len(points)):
        tx = 0.0
        for j in range(1, k + 1):
            if (labels[indices[i, j]] == labels[i]):
                tx += 1
        tx /= k
        txsc[labels[i]] += tx
        nppts[labels[i]] += 1
        txs += tx

    for i in numpy.unique(labels):
        txsc[i] /= nppts[i]
    print(txsc)

    return txs / len(points)


def visualization(x2d, labels, convex_hulls, ellipses, projname, nh):
    import matplotlib as mpl
    points2D_c = []
    unique_labels = numpy.unique(labels)

    for i in unique_labels:
        points2D_c.append(x2d[labels == i, :])
    # Data Visualization
    from matplotlib import cm
    cmap = cm.get_cmap('Dark2')

    from numpy.linalg import linalg
    import numpy as np
    import matplotlib.pyplot as plt
    plt.set_cmap(cmap)
    plt.scatter(x2d[:, 0], x2d[:, 1], c=labels, s=8, edgecolors='none', cmap=cmap, alpha=1.0)
    plt.colorbar(ticks=unique_labels)

    plt.title("2D " + projname + " - NH=" + str(nh * 100.0))
    plt.show()

    vals = [i / unique_labels.shape[0] for i in labels]

    # sp2 = plt.subplot(312)
    # for i in labels:
    #     ch = np.append(convex_hulls[i].vertices, convex_hulls[i].vertices[0])
    #     sp2.plot(points2D_c[i][ch, 0], points2D_c[i][ch, 1], '-', label='$%i$' % i, color=cmap(vals[i]))
    # plt.colorbar(ticks=unique_labels)
    # plt.title(projname + " Convex Hulls")

    def plot_results(X, Y_, means, covariances, index, title, color):
        splot = plt.subplot(1, 1, 1)
        for i, (mean, covar) in enumerate(zip(means, covariances)):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == i):
                continue
            plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color, alpha=0.2)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.6)
            splot.add_artist(ell)
        plt.title(title)

    for i in unique_labels:
        plot_results(x2d[labels == unique_labels[i], :], ellipses[i].predict(x2d[labels == unique_labels[i], :]),
                     ellipses[i].means_,
                     ellipses[i].covariances_, 0, projname + " fitting ellipses", cmap(vals[i]))

    # plt.savefig(projname + ".png", dpi=100)
    plt.show()
