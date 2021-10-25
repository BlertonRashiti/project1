import numpy as np
from numpy import random


## Least squares method with gradient descent

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


## Least squares methode with stochastic gradient descent

def least_squares_SGD(y, tx, w_init, max_iters, gamma):
    
    w = w_init
    N = y.shape[0]
    loss=0
    #print(w)
    for i in range(max_iters):
        
        n = random.randint(N)
        
        e = y[n] - tx[n].dot(w) #error 
        loss = (1/2) * np.transpose(e).dot(e)
        
        gradient =-tx[n]*(y[n]-tx[n].dot(w))
      
  
        #np.ndarray.transpose
        w = w - (gamma*np.transpose([gradient]))
       
    
    return w, loss   

    ## Least squares with normal equations
    
    

    ## Ridge regression with normal equations


    ##Logistic regression with gradient descent 

def logistic_regression(y, tx, initial_w, max_iters, gamma): 

    w = initial_w
    loss = 0
    N = y.shape[0]
    D = w.shape[0]
    grad = np.zeros(D)

    for i in range(max_iters):

        sigmoid = 1/(1+np.exp(-np.dot(tx,w)))
        grad = (np.dot(tx.T,sigmoid-y))/N

        w = w - gamma*grad
    
    loss = np.mean(np.log(1+np.exp(np.dot(tx,w)))-y*np.dot(tx,w))

    return w,loss

    ## Regularized Logistic regression with gradient descent 

def ridge_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):

    w = initial_w
    loss = 0
    N = y.shape[0]
    D = w.shape[0]
    grad = np.zeros(D)

    for i in range(max_iters):

        sigmoid = 1/(1+np.exp(-np.dot(tx,w)))
        grad = (np.dot(tx.T,sigmoid-y))/N + lamba_*w

        w = w - gamma*grad
    
    loss = np.mean(np.log(1+np.exp(np.dot(tx,w)))-y*np.dot(tx,w)) + lamba_/2 * np.dot(w.T,w)

    return w,los
