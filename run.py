from proj1_helpers import *
from implementations import *
import matplotlib.pyplot as plt
import numpy as np

random.seed(29)
# PROCESSING DATA

# Loading data
y, tx, ids = load_csv_data('Data/train.csv')

# Treatement of meaningless values of the 30 covariates

# Subdividision of the problem into 4 sub-problems according to the PRI_jet_num value {0,1,2,3},
# then each sub-problem is divided into 2 depending on whether DER_mass_MMC is defined or not

# Definition of the list of columns containing meaningless values and useless values for the 4 cases
Feature_vector_0 = np.array([4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29])
Feature_vector_1 =  np.array([4, 5, 6, 12, 22, 26, 27, 28])
Feature_vector_2 =  np.array([22]) # Wrong for now
Feature_vector_3 =  np.array([22]) # Wrong for now
ill_defined_feature_tensor = np.array([Feature_vector_0,Feature_vector_1,Feature_vector_2,Feature_vector_3], dtype=object)

# Loading data for the 8 sub-problems, and dimensions of corresponding sub-matrix for each sub-problem
#If subdivision method used
y00, tx00, ids00 = Sub_problem_Generator(y, tx, ids, 0, 0, ill_defined_feature_tensor) #(26123, 17) 
y10, tx10, ids10 = Sub_problem_Generator(y, tx, ids, 1, 0, ill_defined_feature_tensor) #(7562, 21)
y20, tx20, ids20 = Sub_problem_Generator(y, tx, ids, 2, 0, ill_defined_feature_tensor) #(2952, 28)
y30, tx30, ids30 = Sub_problem_Generator(y, tx, ids, 3, 0, ill_defined_feature_tensor) #(1477, 28)
y01, tx01, ids01 = Sub_problem_Generator(y, tx, ids, 0, 1, ill_defined_feature_tensor) #(73790, 18)
y11, tx11, ids11 = Sub_problem_Generator(y, tx, ids, 1, 1, ill_defined_feature_tensor) #(69982, 22)
y21, tx21, ids21 = Sub_problem_Generator(y, tx, ids, 2, 1, ill_defined_feature_tensor) #(47427, 29)
y31, tx31, ids31 = Sub_problem_Generator(y, tx, ids, 3, 1, ill_defined_feature_tensor) #(20687, 29)

# Initially, before dividing the main problem into 8 sub-problems to deal with the meaningless
# values, we replaced the meaningless values by 0, either by the mean of the corresponding
# column or by the median of the corresponding column

#If imputation method used, select one the following three lines
#tx = meaningless_value_median(tx)
#tx = meaningless_value_mean(tx01)
#tx[np.where(tx==-999)] = 0    

# Standardizing columns of features for each design submatrix
tx00 = standardize(tx00)
tx10 = standardize(tx10) 
tx20 = standardize(tx20) 
tx30 = standardize(tx30)  
tx01 = standardize(tx01) 
tx11 = standardize(tx11) 
tx21 = standardize(tx21) 
tx31 = standardize(tx31)

#If impuation method used
#tx=standardize(tx)

# Adding the polynomial term of covariates to obtain a polynomial regression if necessary

#If subdivision method used

tx00_square = np.square(tx00)
tx10_square = np.square(tx10)
tx20_square = np.square(tx20)
tx30_square = np.square(tx30)
tx01_square = np.square(tx01)
tx11_square = np.square(tx11)
tx21_square = np.square(tx21)
tx31_square = np.square(tx31)

tx00=np.concatenate((tx00, tx00_square), axis=1)
tx10=np.concatenate((tx10,  tx10_square), axis=1)
tx20=np.concatenate((tx20,  tx20_square), axis=1)
tx30=np.concatenate((tx30,  tx30_square), axis=1)
tx01=np.concatenate((tx01,  tx01_square), axis=1)
tx11=np.concatenate((tx11, tx11_square), axis=1)
tx21=np.concatenate((tx21, tx21_square), axis=1)
tx31=np.concatenate((tx31,  tx31_square), axis=1)

"""
tx00_3 = np.power(tx00,3)
tx10_3 = np.power(tx10,3)
tx20_3 = np.power(tx20,3)
tx30_3 = np.power(tx30,3)
tx01_3 = np.power(tx01,3)
tx11_3 = np.power(tx11,3)
tx21_3 = np.power(tx21,3)
tx31_3 = np.power(tx31,3)

tx00_4 = np.power(tx00,4)
tx10_4 = np.power(tx10,4)
tx20_4 = np.power(tx20,4)
tx30_4 = np.power(tx30,4)
tx01_4 = np.power(tx01,4)
tx11_4 = np.power(tx11,4)
tx21_4 = np.power(tx21,4)
tx31_4 = np.power(tx31,4)

tx00_5 = np.power(tx00,5)
tx10_5 = np.power(tx10,5)
tx20_5 = np.power(tx20,5)
tx30_5 = np.power(tx30,5)
tx01_5 = np.power(tx01,5)
tx11_5 = np.power(tx11,5)
tx21_5 = np.power(tx21,5)
tx31_5 = np.power(tx31,5)


tx00=np.concatenate((tx00, tx00_square,tx00_3, tx00_4, tx00_5), axis=1)
tx10=np.concatenate((tx10, tx10_square, tx10_3, tx10_4, tx10_5), axis=1)
tx20=np.concatenate((tx20, tx20_square, tx20_3, tx20_4, tx20_5), axis=1)
tx30=np.concatenate((tx30, tx30_square, tx30_3, tx30_4, tx30_5), axis=1)
tx01=np.concatenate((tx01, tx01_square, tx01_3, tx01_4, tx01_5), axis=1)
tx11=np.concatenate((tx11, tx11_square, tx11_3, tx11_4, tx11_5), axis=1)
tx21=np.concatenate((tx21, tx21_square, tx21_3, tx21_4, tx21_5), axis=1)
tx31=np.concatenate((tx31, tx31_square, tx31_3, tx31_4, tx31_5), axis=1)

"""



tx00 = standardize(tx00)
tx10 = standardize(tx10) 
tx20 = standardize(tx20) 
tx30 = standardize(tx30)  
tx01 = standardize(tx01) 
tx11 = standardize(tx11) 
tx21 = standardize(tx21) 
tx31 = standardize(tx31)


#If impuation method used
#tx_square = np.square(tx)
#tx=np.concatenate((tx,tx_square), axis=1)

# Adding an intercept to the model 

#If subdivision method used
tx00 = np.c_[tx00, np.ones(tx00.shape[0])]
tx10 = np.c_[tx10, np.ones(tx10.shape[0])]
tx20 = np.c_[tx20, np.ones(tx20.shape[0])]
tx30 = np.c_[tx30, np.ones(tx30.shape[0])]
tx01 = np.c_[tx01, np.ones(tx01.shape[0])]
tx11 = np.c_[tx11, np.ones(tx11.shape[0])]
tx21 = np.c_[tx21, np.ones(tx21.shape[0])]
tx31 = np.c_[tx31, np.ones(tx31.shape[0])]

#If impuation method used
#tx = np.c_[tx, np.ones(tx.shape[0])]

# We transform the state space of the answer variable Y {1,-1} in {1,0}. {1,-1}->{1,0}
# This adaptation allows a logistic regression, so for OLS method not necessary

#If subdivision method used
y00[np.where(y00 == -1)] = 0
y10[np.where(y10 == -1)] = 0
y20[np.where(y20 == -1)] = 0
y30[np.where(y30 == -1)] = 0
y01[np.where(y01 == -1)] = 0
y11[np.where(y11 == -1)] = 0
y21[np.where(y21 == -1)] = 0
y31[np.where(y31 == -1)] = 0

#If impuation method used
#y[np.where(y == -1)] = 0


# METHOD

# Initializing (hyper)-parameters

#If subdivision method used
w00_init = np.zeros(tx00.shape[1])
w10_init = np.zeros(tx10.shape[1])
w20_init = np.zeros(tx20.shape[1])
w30_init = np.zeros(tx30.shape[1])
w01_init = np.zeros(tx01.shape[1])
w11_init = np.zeros(tx11.shape[1])
w21_init = np.zeros(tx21.shape[1])
w31_init = np.zeros(tx31.shape[1])

#If imputation method used
#w_init = np.zeros(tx.shape[1])

gamma = 0.01
max_iters = 30000

# Model selection via grid search on lambda
#lambda_list=np.arange(0,1,0.05)
lambda_list=[0]

# Errors for each subproblem on train set

#If subdivision method used
errors00=[]
errors10=[]
errors20=[]
errors30=[]
errors01=[]
errors11=[]
errors21=[]
errors31=[]

#If imputation method used
#errors=[]

# Least squares model with stochastic gradient descent estimation

#If subdivision method used
"""
weights00,loss00 = least_squares_SGD(y00, tx00, w00_init, max_iters, gamma)
weights10,loss10 = least_squares_SGD(y10, tx10, w10_init, max_iters, gamma)
weights20,loss20 = least_squares_SGD(y20, tx20, w20_init, max_iters, gamma)
weights30,loss30 = least_squares_SGD(y30, tx30, w30_init, max_iters, gamma)
weights01,loss01 = least_squares_SGD(y01, tx01, w01_init, max_iters, gamma)
weights11,loss11 = least_squares_SGD(y11, tx11, w11_init, max_iters, gamma)
#weights21,loss21 = least_squares_SGD(y21, tx21, w21_init, max_iters, gamma)
#weights31,loss31 = least_squares_SGD(y31, tx31, w31_init, max_iters, gamma)
y_pred00 = predict_labels(weights00,tx00)
y_pred10 = predict_labels(weights10,tx10)
y_pred20 = predict_labels(weights20,tx20)
y_pred30 = predict_labels(weights30,tx30)
y_pred01 = predict_labels(weights01,tx01)
y_pred11 = predict_labels(weights11,tx11)
y_pred21 = predict_labels(weights21,tx21)
y_pred31 = predict_labels(weights31,tx31) """

#If imputation method used
#weights,loss = least_squares_SGD(y, tx, w_init, max_iters, gamma)
#y_pred = predict_labels(weights,tx)

# Regularized logistic regression model with stochastic gradient descent estimation for
# for different hyperparameters lamba_
for l in lambda_list:

    #If subdivision method used
    weights00,loss00 = reg_logistic_regression_SGD(y00, tx00, l, w00_init, max_iters, gamma)
    weights10,loss10 = reg_logistic_regression_SGD(y10, tx10, l, w10_init, max_iters, gamma)
    weights20,loss20 = reg_logistic_regression_SGD(y20, tx20, l, w20_init, max_iters, gamma)
    weights30,loss30 = reg_logistic_regression_SGD(y30, tx30, l, w30_init, max_iters, gamma)
    weights01,loss01 = reg_logistic_regression_SGD(y01, tx01, l, w01_init, max_iters, gamma)
    weights11,loss11 = reg_logistic_regression_SGD(y11, tx11, l, w11_init, max_iters, gamma)
    weights21,loss21 = reg_logistic_regression_SGD(y21, tx21, l, w21_init, max_iters, gamma)
    weights31,loss31 = reg_logistic_regression_SGD(y31, tx31, l, w31_init, max_iters, gamma)    

    y_pred00 = predict_log(weights00,tx00)
    y_pred10 = predict_log(weights10,tx10)
    y_pred20 = predict_log(weights20,tx20)
    y_pred30 = predict_log(weights30,tx30)
    y_pred01 = predict_log(weights01,tx01)
    y_pred11 = predict_log(weights11,tx11)
    y_pred21 = predict_log(weights21,tx21)
    y_pred31 = predict_log(weights31,tx31)

    #If imputation method used
    #weights,loss = re_logistic_SGD(y, tx, w_init, max_iters, gamma)
    #y_pred = predict_log(weights,tx)

    #If subdivision method used
    errors00.append((np.count_nonzero(y00-y_pred00)/tx00.shape[0])*100)
    errors10.append((np.count_nonzero(y10-y_pred10)/tx10.shape[0])*100)
    errors20.append((np.count_nonzero(y20-y_pred20)/tx20.shape[0])*100)
    errors30.append((np.count_nonzero(y30-y_pred30)/tx30.shape[0])*100)
    errors01.append((np.count_nonzero(y01-y_pred01)/tx01.shape[0])*100)
    errors11.append((np.count_nonzero(y11-y_pred11)/tx11.shape[0])*100)
    errors21.append((np.count_nonzero(y21-y_pred21)/tx21.shape[0])*100)
    errors31.append((np.count_nonzero(y31-y_pred31)/tx31.shape[0])*100)

    #If imputation method used
    # errors.append((np.count_nonzero(y-y_pred)/tx.shape[0])*100)

    
errors_list=[errors00, errors10, errors20, errors30, errors01, errors01, errors11, errors21, errors31]
print(errors00, errors10, errors20, errors30, errors01, errors01, errors11, errors21, errors31)
#print((errors00[0]*tx00.shape[0] + errors10[0]*tx10.shape[0] + errors20[0]*tx20.shape[0] + errors30[3]*tx30.shape[0] + errors01[0]*tx01.shape[0]
 #+ errors11[0]*tx11.shape[0] + errors21[0]*tx21.shape[0] + errors31[0]*tx31.shape[0])/(100*tx.shape[0]))


"""
for i in range(1,9):
    plt.subplot(4,2,i) # set the current Axes
    plt.plot(lambda_list, errors_list[i-1], marker='o')
    plt.xlabel("$\lambda$")
    plt.ylabel('Errors in %')

plt.show()
"""
#print(errors)

##############################################################

# Application of the weights obtained via the train set on the test set  

# Loading test set
y2, tx2, ids2 = load_csv_data('Data/test.csv')

# Loading data for the 8 sub-problems, and dimensions of corresponding sub-matrix for each sub-problem
y00_2, tx00_2, ids00_2 = Sub_problem_Generator(y2, tx2, ids2, 0, 0, ill_defined_feature_tensor) #(59263, 17)
y10_2, tx10_2, ids10_2 = Sub_problem_Generator(y2, tx2, ids2, 1, 0, ill_defined_feature_tensor) #(17243, 21)
y20_2, tx20_2, ids20_2 = Sub_problem_Generator(y2, tx2, ids2, 2, 0, ill_defined_feature_tensor) #(6743, 28)
y30_2, tx30_2, ids30_2 = Sub_problem_Generator(y2, tx2, ids2, 3, 0, ill_defined_feature_tensor) #(3239, 28)
y01_2, tx01_2, ids01_2 = Sub_problem_Generator(y2, tx2, ids2, 0, 1, ill_defined_feature_tensor) #(168195, 18)
y11_2, tx11_2, ids11_2 = Sub_problem_Generator(y2, tx2, ids2, 1, 1, ill_defined_feature_tensor) #(158095, 22)
y21_2, tx21_2, ids21_2 = Sub_problem_Generator(y2, tx2, ids2, 2, 1, ill_defined_feature_tensor) #(107905, 29)
y31_2, tx31_2, ids31_2 = Sub_problem_Generator(y2, tx2, ids2, 3, 1, ill_defined_feature_tensor) #(47555, 29)

# Standardizing columns of features for each design sub-matrix
#If subdivison method used
tx00_2 = standardize(tx00_2)
tx10_2 = standardize(tx10_2) 
tx20_2 = standardize(tx20_2) 
tx30_2 = standardize(tx30_2)  
tx01_2 = standardize(tx01_2) 
tx11_2 = standardize(tx11_2) 
tx21_2 = standardize(tx21_2) 
tx31_2 = standardize(tx31_2)

#If impuation method used
#tx2=standardize(tx2)

tx00_square_ = np.square(tx00_2)
tx10_square_ = np.square(tx10_2)
tx20_square_ = np.square(tx20_2)
tx30_square_ = np.square(tx30_2)
tx01_square_ = np.square(tx01_2)
tx11_square_ = np.square(tx11_2)
tx21_square_ = np.square(tx21_2)
tx31_square_ = np.square(tx31_2)


tx00_2=np.concatenate((tx00_2, tx00_square_), axis=1)
tx10_2=np.concatenate((tx10_2,  tx10_square_), axis=1)
tx20_2=np.concatenate((tx20_2,  tx20_square_), axis=1)
tx30_2=np.concatenate((tx30_2,  tx30_square_), axis=1)
tx01_2=np.concatenate((tx01_2,  tx01_square_), axis=1)
tx11_2=np.concatenate((tx11_2, tx11_square_), axis=1)
tx21_2=np.concatenate((tx21_2, tx21_square_), axis=1)
tx31_2=np.concatenate((tx31_2,  tx31_square_), axis=1)


tx00_2 = standardize(tx00_2)
tx10_2 = standardize(tx10_2) 
tx20_2 = standardize(tx20_2) 
tx30_2 = standardize(tx30_2)  
tx01_2 = standardize(tx01_2) 
tx11_2 = standardize(tx11_2) 
tx21_2 = standardize(tx21_2) 
tx31_2 = standardize(tx31_2)


# Adding an intercept to the model 
#If subdivison method used
tx00_2 = np.c_[tx00_2, np.ones(tx00_2.shape[0])]
tx10_2 = np.c_[tx10_2, np.ones(tx10_2.shape[0])]
tx20_2 = np.c_[tx20_2, np.ones(tx20_2.shape[0])]
tx30_2 = np.c_[tx30_2, np.ones(tx30_2.shape[0])]
tx01_2 = np.c_[tx01_2, np.ones(tx01_2.shape[0])]
tx11_2 = np.c_[tx11_2, np.ones(tx11_2.shape[0])]
tx21_2 = np.c_[tx21_2, np.ones(tx21_2.shape[0])]
tx31_2 = np.c_[tx31_2, np.ones(tx31_2.shape[0])]

#If imputation method used
#tx2=np.c_(tx2, np.ones(tx2.shape[0]) )

# Predictions
#If subdivison method used
y_pred00_2 = predict_log(weights00,tx00_2)
y_pred10_2 = predict_log(weights10,tx10_2)
y_pred20_2 = predict_log(weights20,tx20_2)
y_pred30_2 = predict_log(weights30,tx30_2)
y_pred01_2 = predict_log(weights01,tx01_2)
y_pred11_2 = predict_log(weights11,tx11_2)
y_pred21_2 = predict_log(weights21,tx21_2)
y_pred31_2 = predict_log(weights31,tx31_2)

#If imputation method used
#y2_final= predict_log(weights,tx)

# Convert the state space in {-1,1} if the method used is logistic regression
# If subdivision method used
y_pred00_2[np.where(y_pred00_2==0)] = -1
y_pred10_2[np.where(y_pred10_2==0)] = -1
y_pred20_2[np.where(y_pred20_2==0)] = -1
y_pred30_2[np.where(y_pred30_2==0)] = -1
y_pred01_2[np.where(y_pred01_2==0)] = -1
y_pred11_2[np.where(y_pred11_2==0)] = -1
y_pred21_2[np.where(y_pred21_2==0)] = -1
y_pred31_2[np.where(y_pred31_2==0)] = -1
ids2_final = np.concatenate((ids00_2, ids10_2, ids20_2, ids30_2, ids01_2, ids11_2, ids21_2, ids31_2))
y2_final = np.concatenate((y_pred00_2, y_pred10_2, y_pred20_2, y_pred30_2, y_pred01_2, y_pred11_2, y_pred21_2, y_pred31_2))

create_csv_submission(ids2_final, y2_final, "subdividedlogpredictionsok30000.csv")
