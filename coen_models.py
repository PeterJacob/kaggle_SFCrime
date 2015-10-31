__author__ = 'coenjonker'

import pandas as pd
import numpy as np
from model import Model

class MostLikelyClassPerRegion(Model):

    def __init__(self, classes, names):
        super(MostLikelyClassPerRegion, self).__init__(classes, names)
        self.pdid = list(self.names).index('PdDistrict')
        self.trained = False


    def fit(self, train_data, labels):
        classlist = list(self.classes_)
        self.counts = dict()
        for i, X in enumerate(train_data):
            if X[self.pdid] in self.counts:
                self.counts[X[self.pdid]][classlist.index(labels[i])] += 1
            else:
                self.counts[X[self.pdid]] = np.zeros(len(self.classes_))

        self.ratios = dict()
        for k, v in self.counts.items():
            self.ratios[k] = v / np.sum(v)

        self.trained = True

    def predict_point(self, x):
        if not self.trained:
            raise Exception("Not trained yet")
        return self.ratios[x[self.pdid]]

    def predict_proba(self, X):
        return np.apply_along_axis(self.predict_point, 1, X)



class MostLikelyClassPerWeekPerDistrict(MostLikelyClassPerRegion):

    def __init__(self, classes, names):
        super(MostLikelyClassPerWeekPerDistrict, self).__init__(classes, names)


    def fit(self, train_data, labels):
        pass








