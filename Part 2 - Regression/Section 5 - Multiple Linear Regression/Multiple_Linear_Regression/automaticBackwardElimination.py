# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 23:24:11 2018

@author: eherd
"""

#Automatic backward Elimination p-values
import statsmodels.formula.api as sm
import numpy as np

def backwardElimination(x,sl):
    numVars = len(X[0])
    for i in range(0,numVars):
        regressor_OLS = sm.OLS(y,x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if (maxVar>sl):
            for j in range(0, maxVars - i):
                if(regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x,j,1)
    regressor_OLS.summary()
    return x

SL = 0.05
X_opt = X[:,[0,1,2,3,4,5]]
X_Modeled = backwardElimination(X_opt, SL)


def backwarEliminationRSquared(x, SL):
    numVars = len(x[0])
    #depending of the size of x
    temp = np.zeros((50,6)).astype(int)
    for i in range(0,numVars):
        regressor_OLS = sm.OLS(y,x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if(maxVar > SL):
            for j in range(0, numVars - i):
                if(regressor_OLS.pvalues[j] == maxVar):
                    temp[:,j] = x[:,j]
                    x = np.delete(x,j,1)
                    tmp_regressor = sm.OLS(y,x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if(adjR_before >= adjR_after):
                        x_rollback= np.hstack(x,temp[:,[0,j]])
                        x_rollback = np.delete(x_rollback,j,1)
                        print(regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)