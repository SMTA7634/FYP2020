# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 23:21:45 2019

@author: Sofiah
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import joblib
from joblib import load, dump
import time

#INPUT DATA
dfff=pd.read_csv('30MIN_2.csv')# original input data
lowf=pd.read_csv('30MIN2_C15.csv')# convoluted data

R=dfff.drop(['X(t+360)', 'seconds'], axis=1)
s=dfff['X(t+360)']
R_train, R_test, s_train, s_test = train_test_split(R, s, test_size=0.50, random_state=42)
     
#RF MODEL  for original input data
t1 = time.time()
regrO = RandomForestRegressor( n_estimators=250, random_state=7) #O not 0
regrO.fit(R_train,s_train)
predictionO=regrO.predict(R_test)
elapsed1 = time.time() - t1

# RF MODEL PERFORMANCE METRICS

from sklearn.metrics import r2_score
AO=r2_score(s_test, predictionO)
print(AO)

from sklearn.metrics import median_absolute_error
BO= median_absolute_error(s_test, predictionO)
print(BO)

from sklearn.metrics import mean_squared_error
CO=mean_squared_error(s_test,predictionO) 
print(CO)

from sklearn.metrics import mean_absolute_error
DO=mean_absolute_error(s_test, predictionO)
print(DO)

from sklearn.metrics import explained_variance_score
EO=explained_variance_score(s_test, predictionO)
print(EO)


from sklearn.model_selection import cross_val_score
scoresO = cross_val_score(regrO, R, s, cv=3)
print(scoresO)

plt.scatter(s_test,predictionO, color='blue')
plt.xlabel("s_test")
plt.ylabel("predictionO")

##############################################################################################


#SVR MODEL for original input data


# grid search to find gamma and C values



from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}
from sklearn.svm import SVR

grid=GridSearchCV(SVR(),param_grid,verbose=3)

grid.fit(R_train,s_train)
   # Best C and gamma was inserted into the regressor1 SVR model below

from sklearn.svm import SVR
t2 = time.time()
regressor1=SVR(kernel='rbf', C=10, gamma=0.0001)
regressor1.fit(R_train,s_train)
prediction1=regressor1.predict(R_test)
elapsed2 = time.time() - t2
# Performance metrics for SVR
A1=r2_score(s_test,prediction1)
print(A1)

from sklearn.metrics import median_absolute_error
B1=median_absolute_error(s_test,prediction1)
print(B1)

from sklearn.metrics import mean_squared_error
C1=mean_squared_error(s_test,prediction1) 
print(C1)

from sklearn.metrics import mean_absolute_error
D1=mean_absolute_error(s_test, prediction1)
print(D1)

from sklearn.metrics import explained_variance_score
E1=explained_variance_score(s_test, prediction1)
print(E1)
#plt.scatter(y_test,prediction, color='blue')

from sklearn.model_selection import cross_val_score
scores1 = cross_val_score(regressor1, R, s, cv=3)
print(scores1)


plt.scatter(s_test,prediction1, color='blue')
plt.xlabel("s_test")
plt.ylabel("prediction1")

####################################################################################
# SVR + RF MODEL (high freq and low freq input data)

hf=dfff-lowf # high freq data
   
#RF for low freq data (convoluted data)
X=lowf.drop(['X(t+360)', 'seconds'], axis=1)
y=lowf['X(t+360)']
      
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
#SVR for high data
W=hf.drop(['X(t+360)','seconds'], axis=1)
z=hf['X(t+360)']


W_train, W_test, z_train, z_test = train_test_split(W, z, test_size=0.5, random_state=42)
t3 = time.time()
regr2 = RandomForestRegressor( n_estimators=250, random_state=7)
regr2.fit(X_train,y_train)
prediction_rf=regr2.predict(X_test)

#joblib.dump(regr2,'LF30_joblib')   # serializing regr2 model

# grid search to find gamma and C values


from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}
from sklearn.svm import SVR

grid=GridSearchCV(SVR(),param_grid,verbose=3)

grid.fit(W_train,z_train)



regressor3=SVR(kernel='rbf',C=10, gamma=0.0001)
regressor3.fit(W_train,z_train)

prediction_svr=regressor3.predict(W_test)

prediction=prediction_rf+prediction_svr

#joblib.dump(regressor3,'HF30_joblib')  # serializing regressor3 model
elapsed3= time.time() - t3

plt.scatter(s_test,prediction, color='blue')
plt.xlabel("s_test")
plt.ylabel("prediction")
##############################################################################33
  # PERFORMANCE METRICS FOR SVR+RF MODEL     
from sklearn.metrics import r2_score
A=r2_score(s_test,prediction)
print(A)

from sklearn.metrics import median_absolute_error
B=median_absolute_error(s_test,prediction)
print(B)

from sklearn.metrics import mean_squared_error
C=mean_squared_error(s_test,prediction) 
print(C)

from sklearn.metrics import mean_absolute_error
D=mean_absolute_error(s_test, prediction)
print(D)

from sklearn.metrics import explained_variance_score
E=explained_variance_score(s_test, prediction)
print(E)    

#################################################################
#predictions for RF and SVR for input array X_new

X_new=[[-0.009,-0.003,-0.005]]# input array fed into prediction model
ynewO=regrO.predict(X_new)
print(ynewO)

ynew1=regressor1.predict(X_new)
print(ynew1)




########################################################################
# prediction for SVR+RF model for input array X_new
w=2
     
X_new=[-0.009,-0.003,-0.005]

#Define mask and store as an array
mask=np.ones((1,w))/w
mask=mask[0,:]
convolved_data= np.convolve(X_new,mask,'same')
low_freq=convolved_data  #input for rf pred
      
high_freq=X_new-low_freq

high_freq=high_freq.reshape(1,-1)     

low_freq=low_freq.reshape(1,-1)

ynew3=regr2.predict(low_freq)
ynew4=regressor3.predict(high_freq)
      
prediction34=ynew3+ynew4
print(prediction34)