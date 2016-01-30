__author__ = 'coenjonker'

import pandas as pd
import numpy as np
from model import Model
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder

class MostLikelyClassPerRegion(Model):

    def __init__(self, classes, names):
        super(MostLikelyClassPerRegion, self).__init__(classes, names)
        self.pdid = self.getid('PdDistrict')
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
        self.wdid = self.getid('DayOfWeek')


    def fit(self, train_data, labels):
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.models = dict((d, MostLikelyClassPerRegion(self.classes_, self.names)) for d in days)

        for day, model in self.models.items():
            ids = train_data[:, self.wdid] == day
            model.fit(train_data[ids], labels[ids])

    def predict_point(self, x):
        return self.models[x[self.wdid]].predict_point(x)


class SimpleSVM(Model):

    def __init__(self, classes, names):
        super(SimpleSVM, self).__init__(classes, names)

    def encode(self, train_data):
        pass


    def fit(self, train_data, labels):
        pass










