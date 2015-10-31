__author__ = 'coenjonker'

import pandas as pd
import numpy as np
from model import Model

class MostLikelyClassPerRegion(Model):

    def __init__(self, classes, names):
        super(MostLikelyClassPerRegion, self).__init__(classes, names)


    def fit(self, train_data, labels):
        pdid = list(self.names).index('PdDistrict')
        classlist = list(self.classes_)
        self.counts = dict()
        for i, X in enumerate(train_data):
            if X[pdid] in self.counts:
                self.counts[X[pdid]][classlist.index(labels[i])] += 1
            else:
                self.counts[X[pdid]] = np.zeros(len(self.classes_))

        self.ratios = dict()
        for k, v in self.counts:
            self.ratios[k] = v / np.sum(v)

    def predict_proba(self, X):



