# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 23:27:47 2018

@author: eherd
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Encoding categorical data
#encoding the independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,-1] = labelencoder_X.fit_transform(X[:,-1])
onehotencoder = OneHotEncoder(categorical_features=[-1])
X = onehotencoder.fit_transform(X).toarray()
 

#avoiding the dummy variable Trap. Have n-1 dummy variables
X = X[:,1: ]

#splitting the dataset into the training set and the test 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

#Fitting multiple Linear Regressin to the Training set
from sklearn.linear_model import LinearRegression
regressor =  LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
#Adding X to a column of ones. to modify the formula
# y = b0*x0+b1*x1 ... bn*xn
X = np.append(arr= np.ones((50,1)).astype(int) , values = X , axis=1)

#STEP 2
#contains the independent variables that are statiscally significant to the dependent variable
X_opt = X[:,[0,1,2,3,4,5]]
#OLS Ordinary Least Squares
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#To see the p-values
regressor_OLS.summary()

#Removing x2 (second categorical column) predictor with index 2
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Removing x1 (first categorical column) predictor with index 1
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Removing x2 (Administration column) predictor with index 2
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Removing x2 (Marketing column) predictor with index 2
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


