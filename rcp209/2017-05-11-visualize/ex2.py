#!/usr/bin/python
# -*- coding: utf-8

from keras.models import model_from_yaml
def loadModel(savename):
  with open(savename+".yaml", "r") as yaml_file:
    model = model_from_yaml(yaml_file.read())
  print "Yaml Model ",savename,".yaml loaded "
  model.load_weights(savename+".h5")
  print "Weights ",savename,".h5 loaded "
  return model