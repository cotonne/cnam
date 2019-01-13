# coding=utf-8

from sklearn.model_selection import train_test_split

from util import codageDisjonctifComplet


def print_history(history):
    import matplotlib.pyplot as plt
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'loss'], loc='upper left')
    plt.show()


def build(x, y):
    ## hyperparametres
    batch_size = 100
    nb_epoch = 1000
    learning_rate = 0.5

    #  On créé un réseau de neurones vide.
    from keras.models import Sequential

    model = Sequential()

    #  On ajoute des couches avec la fonction add.
    from keras.layers import Dense, Activation, Dropout
    from keras.regularizers import l2
    model.add(Dense(8, input_dim=x.shape[1], name='fc-cache1', kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, input_dim=8, name='fc-cache2', kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, input_dim=8, name='fc-sortie'))
    #  - couche d’activation de type softmax
    model.add(Activation('sigmoid'))

    # visualisation de l'architecture
    model.summary()

    from keras.optimizers import SGD

    sgd = SGD(learning_rate)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'binary_crossentropy'])

    history = model.fit(x, y, validation_split=0.2, batch_size=batch_size, epochs=nb_epoch, verbose=0)

    # print_history(history)

    return model


def build_svm(x, y):
    #### SVC
    tuned_parameters = {
        'C': np.fromiter(map(lambda l: 10 ** (-l), range(-2, 2)), dtype=np.float64),
        'kernel': [ 'rbf', 'sigmoid'], # 'linear', 'poly',
        'degree': range(2, 5),
        'gamma': np.fromiter(map(lambda l: 10 ** (-l), range(-4, 4)), dtype=np.float64)
    }
    from sklearn.model_selection import GridSearchCV
    from sklearn import svm
    clf = GridSearchCV(
        svm.SVC(),
        tuned_parameters,
        # cv=5,
        n_jobs=4,
        verbose=2,
        # scoring=scoring
    )

    clf.fit(x, y)
    # print(clf.cv_results_)

    return clf.best_estimator_


def predict(ind, i):
    y_predict = models[ind].predict(X_test[X_test[cluster_variable] == i].drop(cluster_variable, 1))
    col = np.array(y_predict)
    col_bis = (col >= 0.5).astype(int)
    y_true = Y_test[X_test[cluster_variable] == i]
    from sklearn.metrics import accuracy_score
    return accuracy_score(col_bis, y_true)


import numpy as np
import pandas as pd

from sklearn import preprocessing

filename = 'data_train_ncp_15.csv'
filename_test = "data_test_ncp_15.csv"
csv_res = "python_mlp.csv"
delimiter = ';'

data = []
X = pd.read_csv('missMDA_Regularized.csv', header=0, sep=delimiter, error_bad_lines=False)
# X = X.drop("lvefbin", 1)
X = codageDisjonctifComplet(X, "centre")
X = codageDisjonctifComplet(X, "country")
cols = X.columns
X = preprocessing.StandardScaler().fit_transform(X)

X = pd.DataFrame(X, columns=cols)
y = pd.read_csv(filename, header=0, sep=delimiter, error_bad_lines=False,
                usecols=["lvefbin"],
                dtype={"lvefbin": 'S4'})
y = y.values.ravel()

labels, y = np.unique(y, return_inverse=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=42)
cluster_variable = 'gender'

models = [
    build_svm(X_train[X_train[cluster_variable] == i].drop(cluster_variable, 1), Y_train[X_train[cluster_variable] == i])
    for i in
    np.unique(X_train[cluster_variable])]

print([predict(ind, i) for ind, i in enumerate(np.unique(X_test[cluster_variable]))])

models = [
    build(X_train[X_train[cluster_variable] == i].drop(cluster_variable, 1), Y_train[X_train[cluster_variable] == i])
    for i in
    np.unique(X_train[cluster_variable])]

print([predict(ind, i) for ind, i in enumerate(np.unique(X_test[cluster_variable]))])

#
# PREDICTION
#
# X_clean = pd.read_csv(filename_test, header=0, sep=delimiter, error_bad_lines=False)
# # X_test = preprocessing.StandardScaler().fit_transform(X_clean)
# X_test = X_clean  # pd.DataFrame(X_clean, columns=X_clean.columns)
#
# predict_kmeans = kmeans.predict(X_test)
# X_test['cluster'] = predict_kmeans
#
# col = [models[int(x['cluster'])].predict(x.drop("cluster").to_frame().transpose())[0][0] for index, x in
#        X_test.iterrows()]
# col = np.array(col)
# col_bis = (col >= 0.5).astype(int)
# df = pd.DataFrame(labels[col_bis])
# df.to_csv(csv_res, index=False, encoding='utf-8')
