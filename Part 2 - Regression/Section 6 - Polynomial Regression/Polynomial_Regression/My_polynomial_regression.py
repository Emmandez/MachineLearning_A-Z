# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 17:05:15 2018

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

#fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
#Degree os polynomial features
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X) 

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#Visualising the linear regression results
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X,y, color="red")
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color="blue")
plt.title("Truth of Bluff (Linear Regression results) ")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial regression results
plt.scatter(X,y, color="red")
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color="blue")
plt.title("Truth of Bluff (Polynomial Regression results) ")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear regression
lin_reg.predict(6.5)

#Predicting a new result with polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(6.5))