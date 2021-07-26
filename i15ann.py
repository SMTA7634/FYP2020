# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 21:27:06 2019

@author: Sofiah
"""


import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import time
import matplotlib.pyplot as plt


df=pd.read_csv('30MIN_2.csv')# original data

W=df.drop(['X(t+360)','seconds'], axis=1)
z=df['X(t+360)']
      
W_train, W_test, z_train, z_test = train_test_split(W, z, test_size=0.5, random_state=42)


t1 = time.time()

model = Sequential()
model.add(Dense(3, input_dim=3, init='uniform', activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
# Fit the model
model.fit(W_train, z_train, epochs=100, batch_size=20,  verbose=2)

prediction = model.predict(W_test)
elapsed1 = time.time() - t1
##############################################################################33
plt.scatter(z_test,prediction, color='blue')
plt.xlabel("z_test")
plt.ylabel("prediction")


from sklearn.metrics import r2_score
A=r2_score(z_test,prediction)
print(A)

from sklearn.metrics import median_absolute_error
B=median_absolute_error(z_test,prediction)
print(B)

from sklearn.metrics import mean_squared_error
C=mean_squared_error(z_test,prediction) 
print(C)

from sklearn.metrics import mean_absolute_error
D=mean_absolute_error(z_test, prediction)
print(D)

from sklearn.metrics import explained_variance_score
E=explained_variance_score(z_test, prediction)
print(E)    




    
   
      