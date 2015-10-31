__author__ = 'coenjonker'

from functions import *
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from model import Model
from sklearn.metrics.pairwise import pairwise_distances
import sys

class NearestNeighborTimeSlice(Model):
    def __init__(self, len_window, classes, names, n_neighbors):
        super(NearestNeighborTimeSlice, self).__init__(classes, names)
        self.len_win = len_window
        # self.nn_clf = NearestNeighbors(n_neighbors)
        self.index_dict = dict()
        self.n_neighbors = n_neighbors


    def fit(self, X, y):
        self.X_memory = X
        self.y_memory = y
        weeknumber_index = self.names.tolist().index('WeekNumber')

        for w in xrange(np.max(X[:,weeknumber_index])+1):
            self.index_dict[w] = np.where(np.logical_and(
                w - (self.len_win - 1) / 2 < self.X_memory[:,weeknumber_index],
                self.X_memory[:,weeknumber_index] < w + (self.len_win - 1) / 2
            ))[0]


    def predict(self, X):
        weeknumber_index = self.names.tolist().index('WeekNumber')
        x_index = self.names.tolist().index('X')
        y_index = self.names.tolist().index('Y')
        of_the_jedi = np.empty((len(X), len(self.classes_)))
        # for idx, x in enumerate(X):
        #     weeknum = x[weeknumber_index]
        #     # get all instances within window
        #     indices = self.index_dict[weeknum]
        #     # print(indices)
        #     # print(self.X_memory.shape)
        #
        #     timeslice = self.X_memory[indices]
        #     timeslice = timeslice[:,[x_index, y_index]]
        #
        #     nearest_indices = np.argsort(pairwise_distances(x[[x_index, y_index]], timeslice, n_jobs=8)[0])
        #     nearest_indices = nearest_indices[:len(nearest_indices)*self.n_neighbors]
        #
        #     cnt = Counter(self.y_memory[indices][nearest_indices])
        #
        #     counts = np.array(
        #         [float(cnt[cls]) for cls in self.classes_]
        #     )
        #
        #     probs = counts / np.sum(counts)
        #     of_the_jedi[idx] = probs

        for weeknum in self.index_dict.keys():
            subset_indices = np.where(
                X[:,weeknumber_index]==weeknum
            )[0]


            if len(subset_indices) < 1:
                continue

            memory_indices = self.index_dict[weeknum]
            timeslice = self.X_memory[memory_indices]

            X_sub = X[subset_indices]

            nearest_indices = np.argsort(pairwise_distances(X_sub[:,[x_index, y_index]], timeslice[:,[x_index, y_index]], n_jobs=8))
            # print subset_indices
            # print nearest_indices
            # print nearest_indices.shape

            nearest_indices = nearest_indices[:,:(nearest_indices.shape[1]-1)*self.n_neighbors]

            y_sub = self.y_memory[memory_indices]

            for idx, ss_idx in enumerate(subset_indices):
                cnt = Counter(y_sub[nearest_indices[idx]])
                counts = np.array(
                    [float(cnt[cls]) for cls in self.classes_]
                )
                probs = counts / np.sum(counts)
                of_the_jedi[ss_idx] = probs




        return of_the_jedi