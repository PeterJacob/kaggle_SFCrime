__author__ = 'coenjonker'


import numpy as np
import pandas as pd
import sys
import os

from load_data import load_data

if __name__ == "__main__":
    train_f = sys.argv[1]
    out_dir = sys.argv[2]

    data = load_data(train_f)

    labels = data['Category']
    classes = np.array(pd.unique(data.Category.ravel()))

    data.drop('Category', 1, inplace=True)
    data.drop('Descript', 1, inplace=True)
    data.drop('Resolution', 1, inplace=True)

    names = np.array(data.columns)

    np_train = np.array(data.values)

    np.save(os.path.join(out_dir, 'names.npy'), names)
    np.save(os.path.join(out_dir, 'classes.npy'), classes)
    np.save(os.path.join(out_dir, 'train.npy'), np_train)
    np.save(os.path.join(out_dir, 'labels.npy'), labels)