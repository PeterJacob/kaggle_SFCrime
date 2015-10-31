# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:07:52 2015

@author: coenjonker
"""


import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.cross_validation import StratifiedKFold

from matplotlib import pyplot as plt

data = pd.read_csv('/Users/coenjonker/kaggle/SFCrime/data/train.csv')


targets = data['Category']
dims = data[['X', 'Y']]
folds = 5

neighbor_range = np.arange(15,80,5)
accuracy_neighbors = np.zeros(len(neighbor_range))

stratified = StratifiedKFold(targets, folds)

for j,n in enumerate(neighbor_range):
    print "Testing with {0} neighbors".format(n)
    model = neighbors.KNeighborsClassifier(n_neighbors=n)
    
    scores = np.zeros(folds)
    
    for i, (train, test) in enumerate(stratified):
        print 'fold {0}'.format(i)
        train_data = dims.iloc[train]
        train_labels = targets.iloc[train]
        test_data = dims.iloc[test]
        test_labels = targets.iloc[test]
        model.fit(train_data, train_labels)
        scores[i] = model.score(test_data, test_labels)
    accuracy_neighbors[j] = np.mean(scores)
    print "Average accuracy ({1} neighbors): {0}".format(sum(scores) /float(len(scores)), n)
    
    
plt.plot(neighbor_range, accuracy_neighbors)
plt.show()
    
