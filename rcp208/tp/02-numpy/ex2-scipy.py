# -*- coding: utf-8 -*-

from scipy import stats
from scipy.stats import norm

# size = nombre d'éléments produits, loc = mean, scale = stddev
data = norm.rvs(size=36, loc=1, scale=3)
mat = data.reshape((6,6))

for i in range(0, 6):
    print mat[i, :].mean(), mat[i, :].std()
    print mat[:, i].mean(), mat[:, i].std()

# statistiques descriptives pour geyser

import numpy as np
geyser = np.loadtxt('geyser.txt', dtype=float, delimiter = ' ')
nb_line, nb_col = geyser.shape

for i in range(0, nb_col):
    c = geyser[:, i]
    print c.min(), c.mean(), c.std(), c.max()
    print np.percentile(c, 25), np.median(c), np.percentile(c, 75)
    # aplatissement/kutorsis
    print stats.kurtosis(c)
    # coefficient de variation CV
    print stats.variation(c)


