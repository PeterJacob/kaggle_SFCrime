__author__ = 'coenjonker'

import numpy as np
import pandas as pd

from model import Model


class MostLikelyClassModel(Model):

    def train(self, train_data):
        counts_pr = pd.DataFrame(train_data.groupby('PdDistrict').count()['Dates'])

        counts = pd.DataFrame(train_data.groupby('Category').count()['Dates'])
        counts['Ratio'] = counts / counts.sum()

        self.classes = [x[0] for x in counts['Ratio'].iteritems()]
        self.ratios = [round(x[1], 4) for x in counts['Ratio'].iteritems()]

    def evaluate(self, test_data_point):
        classes_out = ['Id'] + self.classes
        data_out = [test_data_point['Id']] + self.ratios

        return dict(zip(classes_out, data_out))

    def get_classes(self):
        return ['Id'] + self.classes
