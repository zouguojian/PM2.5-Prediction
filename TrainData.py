# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:42:38 2018

@author: Administrator
"""
import pandas as pd
SH_train=pd.read_csv('train_around_weather.csv')
def Dataset():
    return SH_train.loc[:].as_matrix()