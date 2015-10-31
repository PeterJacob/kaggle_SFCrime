__author__ = 'coenjonker'

from functions import *

class WeightedNearestNeighbor():
    def __init__(self, week_weights, classes, names):
        self.week_weights = week_weights
        self.classes_ = classes
        self.names = names

    def fit(self, X, y):
        self.X_memory = X
        self.y_memory = y

    def predict(self, X):
        #magic
        pass



