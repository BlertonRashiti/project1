# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 15:51:48 2021

@author: leagr
"""

import numpy as np
from numpy import random



def least_squares_GD(y, tx, w_init, max_iters, gamma):
    
    #w_init:initial weight vector/ gamma: step-size
    #max_iters: number of steps to run/ lambda: regularization parameter
    # y:Nx1/ tx:NxD/ w:Dx1
    w = w_init
    N = y.shape[0]
    
    for i in range(max_iters):
        
        #L=((1/(2*N)*A*A.transpose()
        
        e = y - tx.dot(w) #error  
        loss = (1/(2*N)) * np.transpose(e).dot(e)
        
        gradient =-(1/N)*tx.dot(y-tx.dot(w))
        w = w - gamma*gradient
       
    return w, loss

        

def least_squares_SGD(y, tx, w_init, max_iters, gamma):
    
    w = w_init
    N = y.shape[0]
    loss=0
    #print(w)
    for i in range(max_iters):
        
        n = random.randint(N-1)
        
        e = y[n] - tx[n].dot(w) #error 
        loss = (1/2) * np.transpose(e).dot(e)
        
        gradient =-tx[n]*(y[n]-tx[n].dot(w))
      
  
        #np.ndarray.transpose
        w = w - (gamma*np.transpose([gradient]))
       

    
    return w, loss   

tx =  np.array([[1,1,0],[0,2,0],[1,0,3]])
y = np.array([[1],[2],[3]])
w_init= np.array(np.random.rand(3,1))
max_iters=500
gamma=0.1


a=least_squares_GD(y, tx, w_init, max_iters, gamma)  
print("least_squares_GD")
print(a)
print("least_squares_SGD")
print (least_squares_SGD(y, tx, w_init, max_iters, gamma)) 



