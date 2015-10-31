def create_folds(weeks=None, n_iter=5, train_size=0.5, test_size=0.5):
    # Create cross-validation shizzle
    from sklearn.cross_validation import StratifiedShuffleSplit
   
    if weeks in None:
        weeks = [1,2,3,1,2,3]
   
    cf = StratifiedShuffleSplit(weeks, n_iter=n_iter, 
                                train_size=train_size, test_size=test_size)
    
    return cf

