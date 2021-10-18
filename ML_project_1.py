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
<<<<<<< HEAD
        gradient =-(1/N) * tx.T.dot(e)
        #print(gradient)
=======
        loss = (1/(2*N)) * np.transpose(e).dot(e)
        
        gradient =-(1/N)*tx.dot(y-tx.dot(w))
>>>>>>> 56909898b84cf8496284fd595ad4476a01513ae5
        w = w - gamma*gradient

    loss = (1/(2*N)) * np.ndarray.transpose(e).dot(e)
    return w, loss

        

def least_squares_SGD(y, tx, w_init, max_iters, gamma):
    
    w = w_init
    N = y.shape[0]
    loss=0
    #print(w)
    for i in range(max_iters):
        
        n = random.randint(N)
        
<<<<<<< HEAD
        e = y[n] - np.dot(tx[n,:],w) #error 
        
        gradient =-tx[n,:]*e
        #print("w apres")
        
        #print((np.array(gradient)))
  
        #np.ndarray.transpose
        w = w - gamma*gradient
        #print(w)
=======
        e = y[n] - tx[n].dot(w) #error 
        loss = (1/2) * np.transpose(e).dot(e)
        
        gradient =-tx[n]*(y[n]-tx[n].dot(w))
      
  
        #np.ndarray.transpose
        w = w - (gamma*np.transpose([gradient]))
       
>>>>>>> 56909898b84cf8496284fd595ad4476a01513ae5

    e = y - tx.dot(w)
    loss = (1/(2*N)) * np.ndarray.transpose(e).dot(e)

    return w,loss

tx =  np.array([[1,1,0],[0,2,1],[1,0,3]])
y = np.array([1,2,3])
w_init= np.array([0,0,0])
print(w_init.shape[0])
print(tx[:,2])
max_iters=100
gamma=0.01

<<<<<<< HEAD
=======
tx =  np.array([[1,1,0],[0,2,0],[1,0,3]])
y = np.array([[1],[2],[3]])
w_init= np.array(np.random.rand(3,1))
max_iters=100
gamma=0.1
>>>>>>> 56909898b84cf8496284fd595ad4476a01513ae5


a=least_squares_GD(y, tx, w_init, max_iters, gamma)  
print("least_squares_GD")
print(a)
print("least_squares_SGD")
print (least_squares_SGD(y, tx, w_init, max_iters, gamma)) 



