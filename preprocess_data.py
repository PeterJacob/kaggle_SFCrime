__author__ = 'coenjonker'


import numpy as np
import pandas as pd
import sys
import os

from load_data import load_data

if __name__ == "__main__":
    train_f = sys.argv[1]
    test_f = sys.argv[2]
    out_dir = sys.argv[3]

    train_data = load_data(train_f)
    test_data = load_data(test_f)

    names = np.array(train_data.columns)
    classes = np.array(pd.unique(train_data.Category.ravel()))

    test_names = np.array(test_data.columns)

    np_train = np.array(train_data.values)
    np_test = np.array(test_data.values)

    np.save(os.path.join(out_dir, 'names.npy'), names)
    np.save(os.path.join(out_dir, 'classes.npy'), classes)
    np.save(os.path.join(out_dir, 'train.npy'), np_train)
    np.save(os.path.join(out_dir, 'test.npy'), np_test)
    np.save(os.path.join(out_dir, 'test_names.npy'), test_names)