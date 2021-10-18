from proj1_helpers import *
from implementations import *
import numpy as np


#p
y, tx, ids = load_csv_data('Data/train.csv', True)
tx[np.where(tx==-999)] = 0 ## the meaningless values are removed from our model, so equal to 0
tx=standardize(tx) #standardize the data, mean of each dimension for features equal to 0 and variance of each dimension for features equal to 1
print(tx[1])
print(tx.shape)



w_init = np.random.rand(30)
gamma = 0.01
max_iters = 10000

#print(logistic_regression(y, tx, w_init, max_iters,gamma))
print(least_squares_GD(y, tx, w_init, max_iters, 0.001))
print(w_init)