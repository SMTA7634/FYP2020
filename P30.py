# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 20:14:22 2019

@author: Admin
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import pickle
import joblib
from joblib import dump, load
lf30=joblib.load('LF30_joblib')
hf30=joblib.load('HF30_joblib')
#mj.predict() #this is regr

class min_30 (object):
    def __init__(self, X1, X2, X3):
    
     
      w=2
     
      DF=[X1,X2,X3]

#Define mask and store as an array
      mask=np.ones((1,w))/w
      mask=mask[0,:]
      convolved_data= np.convolve(DF,mask,'same')
      low_freq=convolved_data  #input for rf pred
      
      high_freq=DF-low_freq #input for svr pred
      
      

      low_freq=low_freq.reshape(1,-1)
      
      
      ynew1=lf30.predict(low_freq)#maybe mj.pred no longer regr
      

      
      high_freq=high_freq.reshape(1,-1)     
      ynew2=hf30.predict(high_freq)
      prediction=ynew1+ynew2

    
    #def predict_30MIN (self):
      print(prediction)
      # -*- coding: utf-8 -*-


