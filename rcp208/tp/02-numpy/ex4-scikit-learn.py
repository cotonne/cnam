from sklearn import preprocessing
import numpy as np

geyser = np.loadtxt('geyser.txt')
print geyser[:5, :]

geyserNormalise = preprocessing.scale(geyser)
print geyserNormalise[:5, :]

print "mean " , geyser.mean(axis=0)
print "mean norm " , geyserNormalise.mean(axis=0)
print "std ", geyser.std(axis = 0)
print "std norm " , geyserNormalise.std(axis = 0)
