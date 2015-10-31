__author__ = 'coenjonker'

import pandas as pd
import numpy as np
from model import Model

class MostLikelyClassPerRegion(Model):

    def __init__(self, classes, names):
        super(MostLikelyClassPerRegion, self).__init__(classes, names)

    def processRatio(self, r):
        if r < 0.01:
            return 0
        return round(r, 4)

    def fit(self, data):
        counts_district = pd.DataFrame(data.groupby('PdDistrict').count()['Dates'])
        count_dict = dict([x for x in counts_district['Dates'].iteritems()])
        regions = count_dict.keys()
        self.region_ratios = dict()

        for region in regions:
            region_data = data.query("PdDistrict=='{0}'".format(region))
            counts = pd.DataFrame(region_data.groupby('Category').count()['Dates'])
            counts['Ratio'] = counts / counts.sum()
            self.region_ratios[region] = dict(
                [(x[0], self.processRatio(x[1])) for x in counts['Ratio'].iteritems()]
            )

    def evaluate(self, test_data_point):
        print test_data_point
        this_region_ratios = self.region_ratios[test_data_point[3]]

        for c in self.classes_:
            if c not in this_region_ratios:
                this_region_ratios[c] = 0.0
        return this_region_ratios

    def predict_proba(self, X):
        result = np.empty((np.shape(X)[0], len(self.classes)))
        for i, point in enumerate(X):
            print point
            result[i] = self.evaluate(point)




