import numpy as np
from numpy import random

# Definition of general parameters of functions
# w_init:initial weight vector/ gamma: step-size (which wil be defined as fixed)
# max_iters: number of steps to run/ lambda: regularization parameter
# y: answer variable with dimension Nx1/ tx: design matrix with dimension NxD
# w: parameters of our model with dimension Dx1

## Least squares method with gradient descent

def least_squares_GD(y, tx, w_init, max_iters, gamma):
    
    w = w_init
    N = y.shape[0]

    for i in range(max_iters):
        e = y - tx.dot(w) #error 
        gradient =-(1/N) * tx.T.dot(e)
        w = w - gamma*gradient

    loss = (1/(2*N)) * np.ndarray.transpose(e).dot(e)

    return w, loss


## Least squares methode with stochastic gradient descent

def least_squares_SGD(y, tx, w_init, max_iters, gamma):
    
    w = w_init
    N = y.shape[0]

    for i in range(max_iters):
        n = random.randint(N-1) # sampling one element uniformly with replacement 
        e = y[n] - np.dot(tx[n,:],w) #error 
        gradient =-tx[n,:]*e
        w = w - gamma*gradient

    e = y - tx.dot(w)
    loss = (1/(2*N)) * np.ndarray.transpose(e).dot(e)

    return w,loss


    ## Ridge regression with normal equations

    def ridge_regression(y, tx, lambda_):
        A = np.ndarray.transpose(tx)
        w = ((np.linalg.inv(tx.dot(A)+lambda_*np.identity(np.size(y)))).dot(tx)).dot(y)
        e = y - tx.dot(w)
        loss = (1/(2*N)) * np.ndarray.transpose(e).dot(e)
        return w , loss

    ## Least squares with normal equations

    def least_squares(y, tx):
        return ridge_regression(y,tx,0)


    ## Regularized Logistic regression with gradient descent 

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):

    w = initial_w
    loss = 0
    N = y.shape[0]
    D = w.shape[0]
    grad = np.zeros(D)

    for i in range(max_iters):

        sigmoid = 1/(1+np.exp(-np.dot(tx,w)))
        grad = (np.dot(tx.T,sigmoid-y))/N + lambda_ * w
        w = w - gamma*grad
    
    loss = np.mean(np.log(1+np.exp(np.dot(tx,w)))-y*np.dot(tx,w)) + lambda_ /2 * np.dot(w.T,w)

    return w,loss

def logistic_regression(y, tx, initial_w, max_iters, gamma): 
    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma)


    ## Regularized logistic regression with stochastic gradient descent algorithm

def reg_logistic_regression_SGD(y, tx, lambda_, initial_w, max_iters, gamma):

    w = initial_w
    loss = 0
    N = y.shape[0]
    D = w.shape[0]
    grad = np.zeros(D)

    for i in range(max_iters):

        n = random.randint(N-1) # sampling one element uniformly with replacement
        e = y[n] - np.dot(tx[n,:],w)
        sigmoid = 1/(1+np.exp(-np.dot(tx[n,:],w)))
        grad = (sigmoid-y[n])*tx[n,:] + lambda_ * w
        w = w - gamma*grad
        
    loss = np.mean(np.log(1+np.exp(np.dot(tx,w))) - y*np.dot(tx,w)) + lambda_ /2 * np.dot(w.T,w)    

    return w,loss

def logistic_regression_SGD(y, tx, initial_w, max_iters, gamma): 
    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma)
