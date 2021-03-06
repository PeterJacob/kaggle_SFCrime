__author__ = 'coenjonker'


class Model(object):

    def __init__(self, classes, names):
        self.classes_ = classes
        self.names = names

    def fit(self, train_data, labels):
        raise NotImplementedError("fit not implemented in model {0}".format(self.__class__))

    def predict_proba(self, X):
        raise NotImplementedError("predict_proba not implemented in model {0}".format(self.__class__))

    def getid(self, name):
        return list(self.names).index(name)

    def get_classes(self):
        """
        Returns list of possible output classes and case Id
        :return:
        """
        return ['Id'] + self.classes_
