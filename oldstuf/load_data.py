# -*- coding: utf-8 -*-

import pandas as pd
from weekcalc import WeekCalculator



def load_data(f):
    # Create a week calculator class
    wc = WeekCalculator()
    
    # load the data
    df = pd.read_csv(f)
    
    # Add weeknumbers
    df['WeekNumber'] = df['Dates'].apply(wc.calc_week)
    
    # return the data frame\
    return df
	
