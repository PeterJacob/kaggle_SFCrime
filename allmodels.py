__author__ = 'coenjonker'

from functions import *
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from model import Model
from sklearn.metrics.pairwise import pairwise_distances
from functools import partial

from multiprocessing import Pool
import sys

def predict_proba_subset((subset_indices, X_sub, timeslice, y_sub), classes, neighbor_frac):
    retval = np.empty((len(X_sub), len(classes)))

    nearest_indices = np.argsort(pairwise_distances(X_sub, timeslice))
    nearest_indices = nearest_indices[:,:(nearest_indices.shape[1]-1)*neighbor_frac]

    for idx, ss_idx in enumerate(subset_indices):
            cnt = Counter(y_sub[nearest_indices[idx]])
            counts = np.array(
                [float(cnt[cls]) for cls in classes]
            )
            probs = counts / np.sum(counts)
            retval[idx] = probs

    return subset_indices, retval

class NearestNeighborTimeSlice(Model):
    def __init__(self, len_window, classes, names, n_neighbors):
        super(NearestNeighborTimeSlice, self).__init__(classes, names)
        self.len_win = len_window
        # self.nn_clf = NearestNeighbors(n_neighbors)
        self.index_dict = dict()
        self.n_neighbors = n_neighbors


    def fit(self, X, y):
        print("fitting")
        self.X_memory = X
        self.y_memory = y
        weeknumber_index = self.names.tolist().index('WeekNumber')

        for w in xrange(np.max(X[:,weeknumber_index])+1):
            self.index_dict[w] = np.where(np.logical_and(
                w - (self.len_win - 1) / 2 < self.X_memory[:,weeknumber_index],
                self.X_memory[:,weeknumber_index] < w + (self.len_win - 1) / 2
            ))[0]

    def slice_data_set(self, X):
        x_index = self.names.tolist().index('X')
        y_index = self.names.tolist().index('Y')
        weeknumber_index = self.names.tolist().index('WeekNumber')


        for weeknum in self.index_dict.keys():
            subset_indices = np.where(
                X[:,weeknumber_index]==weeknum
            )[0]

            if len(subset_indices) < 1:
                continue

            memory_indices = self.index_dict[weeknum]
            timeslice = self.X_memory[memory_indices][:,[x_index, y_index]]
            X_sub = X[subset_indices][:,[x_index, y_index]]

            yield subset_indices, X_sub, timeslice, self.y_memory[memory_indices]

    def predict(self, X):
        print("predicting")

        of_the_jedi = np.empty((len(X), len(self.classes_)))

        responses =  Pool(8).imap_unordered(partial(predict_proba_subset, classes=self.classes_, neighbor_frac=self.n_neighbors), self.slice_data_set(X))
        for indices, probs in responses:
            of_the_jedi[indices] = probs

        return of_the_jedi


    # def predict(self, X):
    #     weeknumber_index = self.names.tolist().index('WeekNumber')
    #     x_index = self.names.tolist().index('X')
    #     y_index = self.names.tolist().index('Y')
    #     of_the_jedi = np.empty((len(X), len(self.classes_)))
    #
    #
    #     for weeknum in self.index_dict.keys():
    #         subset_indices = np.where(
    #             X[:,weeknumber_index]==weeknum
    #         )[0]
    #
    #
    #         if len(subset_indices) < 1:
    #             continue
    #
    #         memory_indices = self.index_dict[weeknum]
    #         timeslice = self.X_memory[memory_indices]
    #
    #         X_sub = X[subset_indices]
    #
    #         nearest_indices = np.argsort(pairwise_distances(X_sub[:,[x_index, y_index]], timeslice[:,[x_index, y_index]], n_jobs=8))
    #         # print subset_indices
    #         # print nearest_indices
    #         # print nearest_indices.shape
    #
    #         nearest_indices = nearest_indices[:,:(nearest_indices.shape[1]-1)*self.n_neighbors]
    #
    #         y_sub = self.y_memory[memory_indices]
    #
    #         for idx, ss_idx in enumerate(subset_indices):
    #             cnt = Counter(y_sub[nearest_indices[idx]])
    #             counts = np.array(
    #                 [float(cnt[cls]) for cls in self.classes_]
    #             )
    #             probs = counts / np.sum(counts)
    #             of_the_jedi[ss_idx] = probs
    #
    #
    #
    #
    #     return of_the_jedi