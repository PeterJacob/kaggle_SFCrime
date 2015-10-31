__author__ = 'coenjonker'

import sys
from coen_models import MostLikelyClassPerRegion
from load_data import load_data

#TODO: Iets slims met numpy arrays om efficient alle modellen te evalueren volgens logloss functie
# We kunnen dit ook lekker laten zitten en KAggle laten evalueren, maar we hebben max 5 inzendingen per dag, dus als we gaan
# Monte Carlo optimaliseren is het wellicht handiger dit ding wel ff te implementeren :)
if __name__ == "__main__":
    train_f = sys.argv[1]
    test_f = sys.argv[2]

    train_data = load_data(train_f)
    test_data = load_data(test_f)

    model = MostLikelyClassPerRegion()

    model.fit(train_data[0:200])
    small_test = test_data[0:10]
    print model.predict_proba(small_test.values)








