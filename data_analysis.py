# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 17:55:16 2021

@author: leagr
"""
import numpy as np

def undefined_value(X):
    "replaces undefined value -999 by the mean of the feature "
    
    Y=X[X!=-999]
    Y_mean=np.mean(Y)
    
    X=np.where(X!=-999, X, Y_mean)
   
    return X
        
a= np.array([[1],[2],[3], [-999]])       
print(undefined_value(a))       
 


def standardize(X):
    " used to standardize a feature of the dataset "
        
    X_mean=np.mean(X, axis=0)
    X_std=np.std(X, axis=0)
    X=(X - X_mean)/X_std
    
    return X


def normalize(X):
    " used to normalize a feature of the dataset "
    X_min=np.min(X)
    X_max=np.max(X)
    X= (X-X_min)/(X_max-X_min)
    
    return X
 
    
def expand_vector(X, degree):
    "returns polynomial expansion of a vector x with degrees ranging from 1 to 'degree'"
    poly_exp = X
    for i in range(2, degree+1):
        poly_exp = np.c_[poly_exp, np.power(X, i)] #concatenation
    return poly_exp

print(expand_vector(a, 3))
