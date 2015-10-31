# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.cross_validation import StratifiedKFold

from matplotlib import pyplot as plt

data = pd.read_csv('/Users/coenjonker/Google Drive/kaggle/SFCrime/data/train.csv')

test = pd.read_csv('/Users/coenjonker/Google Drive/kaggle/SFCrime/data/test.csv')
targets = data['Category']
dims = data[['X', 'Y']]

test_data = test[['X','Y']]
model = neighbors.KNeighborsClassifier(n_neighbors=20)

model.fit(dims, targets)




