import numpy as np
import sklearn.dummy
from PredictionMachine import PredictionMachine

dtree = sklearn.dummy.DummyClassifier(strategy='most_frequent')
PM = PredictionMachine(dtree)
PM.run()
