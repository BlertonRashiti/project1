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
        e = y - tx.dot(w) #error 
        print(e) 
        gradient =-(1/N) * tx.T.dot(e)
        print(gradient)
        w = w - gamma*gradient

    loss = (1/(2*N)) * np.ndarray.transpose(e).dot(e)
    return w, loss


## Least squares methode with stochastic gradient descent

def least_squares_SGD(y, tx, w_init, max_iters, gamma):
    
    w = w_init
    N = y.shape[0]
    
    for i in range(max_iters):

        n = random.randint(N-1)
        
        e = y[n] - np.dot(tx[n,:],w) #error 
        gradient =-tx[n,:]*e
        w = w - gamma*gradient

    e = y - tx.dot(w)
    loss = (1/(2*N)) * np.ndarray.transpose(e).dot(e)

    return w,loss

    
    ## Ridge regression with normal equations

    def ridge_regression(y, tx, Lambda):
    A = np.ndarray.transpose(tx)
        return ((np.linalg.inv(tx.dot(A)+Lambda*np.identity(np.size(y)))).dot(tx)).dot(y)

    ## Least squares with normal equations

    def least_squares(y, tx):
        return ridge_regression(y,tx,0)

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
