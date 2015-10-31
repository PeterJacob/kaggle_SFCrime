__author__ = 'coenjonker'

from functions import *
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from model import Model
from sklearn.metrics.pairwise import pairwise_distances_argmin

class NearestNeighborTimeSlice(Model):
    def __init__(self, len_window, classes, names, n_neighbors):
        super(NearestNeighborTimeSlice, self).__init__(classes, names)
        self.len_win = len_window
        self.nn_clf = NearestNeighbors(n_neighbors)
        self.index_dict = dict()


    def fit(self, X, y):
        self.X_memory = X
        self.y_memory = y
        weeknumber_index = self.names.tolist().index('WeekNumber')

        for w in np.unique(X[:,weeknumber_index]):
            self.index_dict[w] = np.where(np.logical_and(
                w - (self.len_win - 1) / 2 < self.X_memory[:,weeknumber_index],
                self.X_memory[:,weeknumber_index] < w + (self.len_win - 1) / 2
            ))[0]


    def predict(self, X):
        weeknumber_index = self.names.tolist().index('WeekNumber')
        x_index = self.names.tolist().index('X')
        y_index = self.names.tolist().index('Y')
        of_the_jedi = np.empty((len(X), len(self.classes_)))
        for idx, x in enumerate(X):
            weeknum = x[weeknumber_index]
            # get all instances within window
            indices = self.index_dict[weeknum]
            # print(indices)
            # print(self.X_memory.shape)

            timeslice = self.X_memory[indices]
            timeslice = timeslice[:,[x_index, y_index]]

            self.nn_clf.fit(timeslice, self.y_memory[indices])
            _, nearest_indices = self.nn_clf.kneighbors([x[[x_index, y_index]]])
            cnt = Counter(self.y_memory[indices][nearest_indices][0])

            counts = np.array(
                [float(cnt[cls]) for cls in self.classes_]
            )

            probs = counts / np.sum(counts)
            of_the_jedi[idx] = probs

        return of_the_jedi