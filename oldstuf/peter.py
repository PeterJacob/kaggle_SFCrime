def create_folds(strat_col=None, n_iter=5, train_size=0.5, test_size=0.5):
    # Create cross-validation shizzle
    from sklearn.cross_validation import StratifiedShuffleSplit
   
    if strat_col is None:
        strat_col = [1,2,3,1,2,3]
        print('No strat_col given. Using: {}'.format(str(strat_col)))
    
    cf = StratifiedShuffleSplit(strat_col, n_iter=n_iter, 
                                train_size=train_size, test_size=test_size)
    
    return cf

