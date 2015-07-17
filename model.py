__author__ = 'coenjonker'


class Model(object):
    def train(self, train_data):
        raise NotImplementedError("Train function not implemented in model {0}".format(self.__class__))

    def evaluate(self, test_data_point):
        """
        Should return dict with category: probability for data point
        :param test_data_point:
        :return:
        """
        return dict()

    def get_classes(self):
        """
        Returns list of possible output classes
        :return:
        """
        return []
