__author__ = 'coenjonker'

import pandas as pd

from primitive.model import Model


class MostLikelyClassModel(Model):

    def train(self, train_data):
        counts = pd.DataFrame(train_data.groupby('Category').count()['Dates'])
        counts['Ratio'] = counts / counts.sum()

        self.classes = [x[0] for x in counts['Ratio'].iteritems()]
        self.ratios = [round(x[1], 4) for x in counts['Ratio'].iteritems()]

    def evaluate(self, test_data_point):
        data_out = [test_data_point['Id']] + self.ratios

        return dict(zip(self.get_classes(), data_out))


class MostLikeClassPerRegionModel(MostLikelyClassModel):

    def processRatio(self, r):
        if r < 0.01:
            return 0
        return round(r, 4)

    def train(self, train_data):
        super(MostLikeClassPerRegionModel, self).train(train_data)

        counts_district = pd.DataFrame(train_data.groupby('PdDistrict').count()['Dates'])
        count_dict = dict([x for x in counts_district['Dates'].iteritems()])
        self.ratiodict = dict(zip(self.classes, self.ratios))

        regions = count_dict.keys()

        self.region_ratios = dict()

        for region in regions:
            region_data = train_data.query("PdDistrict=='{0}'".format(region))
            counts = pd.DataFrame(region_data.groupby('Category').count()['Dates'])
            counts['Ratio'] = counts / counts.sum()
            self.region_ratios[region] = dict(
                [(x[0], self.processRatio(x[1])) for x in counts['Ratio'].iteritems()]
            )

    def evaluate(self, test_data_point):
        this_region_ratios = self.region_ratios[test_data_point['PdDistrict']]
        this_region_ratios['Id'] = test_data_point['Id']

        for c in self.classes:
            if c not in this_region_ratios:
                this_region_ratios[c] = self.ratiodict[c]


        return this_region_ratios







