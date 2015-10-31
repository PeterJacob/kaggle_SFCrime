__author__ = 'coenjonker'

import inspect

from primitive import allmodels

#TODO: Iets slims met numpy arrays om efficient alle modellen te evalueren volgens logloss functie
# We kunnen dit ook lekker laten zitten en KAggle laten evalueren, maar we hebben max 5 inzendingen per dag, dus als we gaan
# Monte Carlo optimaliseren is het wellicht handiger dit ding wel ff te implementeren :)
if __name__ == __main__:

    models = [name for name, obj in inspect.getmembers(allmodels) if inspect.isclass(obj)]







