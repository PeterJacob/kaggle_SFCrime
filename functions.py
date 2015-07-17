import pandas as pd

__author__ = 'coenjonker'

import csv


def read_data(filename):
    """
    Useless function. Perhaps do some prepping here later
    :param filename:
    :return:
    """
    return pd.read_csv(filename)


def write_data(filename, model, test_data_file):
    """
    Reads the test data, get's a class prob for all classes in all data points
    and writes the result to a file
    :param filename:
    :param model:
    :param test_data:
    :return:
    """

    with open(filename, 'w') as out_obj:
        writer = csv.DictWriter(out_obj, restval=0, fieldnames=model.get_classes())

        writer.writeheader()

        with open(test_data_file, 'r') as in_obj:
            reader = csv.DictReader(in_obj)
            for point in reader:
                writer.writerow(model.evaluate(point))
