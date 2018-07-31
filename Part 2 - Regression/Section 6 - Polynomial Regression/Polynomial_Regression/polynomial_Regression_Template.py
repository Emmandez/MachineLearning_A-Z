# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 22:08:40 2018

@author: eherd
""" 

#Polynomial regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
#X has to be a matrix 10,1 matrix in this case
X = dataset.iloc[:,1:2].values
#Y has to be a vector 
y = dataset.iloc[:,2].values


#Fitting regression Model to the dataset
y_pred = regressor.predict(6.5)

#Visualising the Regression results
plt.scatter(X,y, color="red")
plt.plot(X, regressor.predict(X), color="blue")
plt.title("Truth of Bluff (Regression Model) ")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear regression (higher resolution)
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X.grid.reshape((len(X_grid),1))
plt.scatter(X,y, color="red")
plt.plot(X, regressor.predict(X_grid), color="blue")
plt.title("Truth of Bluff (Regression Model) ")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
