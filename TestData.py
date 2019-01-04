# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:41:27 2018

@author: Administrator
"""
import pandas as pd
SH_test=pd.read_csv('test_around_weather.csv')
def Dataset():
    return SH_test.loc[:].as_matrix()
#SH=Dataset()
#print(SH.shape)