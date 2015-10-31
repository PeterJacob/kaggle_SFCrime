__author__ = 'coenjonker'

from functions import *
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter

class NearestNeighborTimeSlice:
    def __init__(self, len_window, classes, names, n_neighbors):
        self.len_win = len_window
        self.classes_ = classes
        expected_names = [u'Id', u'Dates', u'DayOfWeek', u'PdDistrict', u'Address', u'X', u'Y', 'WeekNumber']
        self.names = names
        if not np.all(np.in1d(expected_names, names)):
            raise Exception("Not all expected variables supplied")
        self.nn_clf = NearestNeighbors(n_neighbors)


    def fit(self, X, y):
        self.X_memory = X
        self.y_memory = y

    def predict(self, X):
        weeknumber_index = self.names.index('WeekNumber')
        x_index = self.names.index('X')
        y_index = self.names.index('Y')
        of_the_jedi = np.empty((len(X), len(self.classes_)))
        for idx, x in enumerate(X):
            weeknum = x[weeknumber_index]
            # get all instances within window
            indices = np.logical_and(
                weeknum - (self.len_win - 1) / 2 < self.X_memory[weeknumber_index],
                self.X_memory[weeknumber_index] < weeknum - (self.len_win - 1) / 2
            )

            timeslice = self.X_memory[indices, [x_index, y_index]]

            self.nn_clf.fit(timeslice, self.y_memory[indices])
            _, nearest_indices = self.nn_clf.kneighbors([x[[x_index, y_index]]])
            cnt = Counter(self.y_memory[indices][nearest_indices])

            counts = np.array(
                [float(cnt[cls]) for cls in self.classes_]
            )

            probs = counts / np.sum(counts)

            of_the_jedi[idx] = probs
        return of_the_jedi




