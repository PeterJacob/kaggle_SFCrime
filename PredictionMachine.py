import numpy as np
import sklearn.preprocessing
import sklearn.cross_validation
import sklearn.metrics

class PredictionMachine(object):
    # This class reads X feature matrix and Y target vector in
    # Creates an 5x2 fold on the dataset
    # Trains and runs a classifier supplied in the init
    # Returns the error
    
    def __init__(self, classifier):
        self.cls = classifier
        self.X_path = '/home/peter/Downloads/X_sample.npy'
        self.Y_path = '/home/peter/Downloads/Y_sample.npy'
        self.X = None
        self.Y = None
            
    def run(self):
        self.read_data()
        cf = self.create_folds()
        
        for train_idx, test_idx in cf:
            X_train, Y_train = self.X[train_idx], self.Y[train_idx]
            X_test,  Y_test  = self.X[test_idx], self.Y[test_idx]
            
            cls.fit(X_train, Y_train)
            Y_prob = cls.predict_proba(X_test)
            
            self.print_error(Y_prob, Y_test, cls.classes_)
    
    def read_data(self):
        self.X = np.load(self.X_path)
        self.Y = np.load(self.Y_path)
    
    def create_folds(self):
        # 5x2 folds
        print self.X.shape
        folds = sklearn.cross_validation.ShuffleSplit(
             n=self.X.shape[0], n_iter=5, train_size=0.5, test_size=0.5)
        
        return folds

    def print_error(self, Y_prob, Y_test, classes_):
        lb = sklearn.preprocessing.LabelBinarizer()
        lb.classes_ = classes_
        Y_test_matrix = lb.transform(Y_test)
        
        ll = sklearn.metrics.log_loss(Y_test_matrix, Y_prob)
        print ll
