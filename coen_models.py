__author__ = 'coenjonker'

import pandas as pd
import numpy as np
from model import Model

class MostLikelyClassPerRegion(Model):

    def __init__(self, classes, names):
        super(MostLikelyClassPerRegion, self).__init__(classes, names)


    def fit(self, train_data):
