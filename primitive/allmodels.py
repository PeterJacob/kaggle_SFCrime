__author__ = 'coenjonker'

from functions import *
import numpy as np

class WeightedNearestNeighbor():
    def __init__(self, week_weights, classes, names):
        self.week_weights = week_weights
        self.classes_ = classes
        expected_names = [u'Id', u'Dates', u'DayOfWeek', u'PdDistrict', u'Address', u'X', u'Y']
        self.names = names
        if not np.all(np.in1d(expected_names, names)):
            raise Exception("Not all expected variables supplied")


    def fit(self, X, y):
        self.X_memory = X
        self.y_memory = y

    def predict(self, X):

        pass



