import numpy as np
import sklearn.dummy
from PredictionMachine import PredictionMachine

features_path = '/home/peter/Documents/kaggle_SFCrime/data/train.npy'
labels_path   = '/home/peter/Documents/kaggle_SFCrime/data/labels.npy'

classifier = sklearn.dummy.DummyClassifier(strategy='most_frequent')
PM = PredictionMachine(classifier, 
    features_path=features_path, labels_path=labels_path)
PM.run()
