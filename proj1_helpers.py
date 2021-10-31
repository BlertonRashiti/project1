# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::1000]
        input_data = input_data[::1000]
        ids = ids[::1000]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


############################################################################


def standardize(X):
    
    centered_data = X - np.mean(X, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    
    return std_data


def meaningless_value_mean(X):
    # replaces undefined value -999 by the mean of the features in design matrix X
    Z = np.copy(X)
    Z[np.where(Z==-999)]=0
    column_means= Z.mean(axis=0)
    X=np.where(X!=-999, X, column_means)

    return X

def meaningless_value_median(X):
    # replaces undefined value -999 by the median of the features in design matrix X
    Z = np.copy(X)
    Z[np.where(Z==-999)]=0
    column_means= np.median(Z,axis=0)
    X=np.where(X!=-999, X, column_means)

    return X

def predict_log(weights, data):
    # Classify the answer variable for logistic regression given weights 
    y_pred = 1/(1+np.exp(-np.dot(data,weights)))
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred


    # Data sub-division

    # Function that splits data into 8 sub problems -> defined by a jet number (PRI_jet_num) and "mass bool" (whether or not Der_mass_MMC is defined)
    # For a given sub-problem (defined by a jet number and a mass bool), there are features which are ill defined and are therfore removed
    # The problem is thus actually a superposition of 8 sub-problems each to be treated seperately
    # Function that takes the total problem and yields the sub-problem defined by a jet number and a "mass bool"
    # For practicity the bool_mass uses 0,1 to use it as an index in "ill_defined_feature_tensor"

def Sub_array_Generator(y, tx, ids, jet_nbr, bool_mass):

    if bool_mass == 1: # finds indices of the sub-problem defined by (jet number, bool mass = 1)
        bool_location = np.where((tx[:,22] == jet_nbr) & (tx[:,0] != -999), True, False)       
    else: # finds indices of the sub-problem defined by (jet number, bool mass = 0)
        bool_location = np.where((tx[:,22] == jet_nbr) & (tx[:,0] == -999), True, False)
    y_sub = y[bool_location]
    tx_sub = tx[bool_location, :]
    ids_sub = ids[bool_location]
    return y_sub, tx_sub, ids_sub

# Function that removes undefined columns

def Feature_remover(tx_sub, jet_nbr, bool_mass, ill_defined_feature_tensor):
    if bool_mass == 0:
        id_keep = np.delete(np.arange(30), np.concatenate(([0], ill_defined_feature_tensor[jet_nbr]), axis=None))
    else:
        id_keep = np.delete(np.arange(30), ill_defined_feature_tensor[jet_nbr])        
    tx_sub_new = tx_sub[:,id_keep]
    return tx_sub_new

# Final Function giving a sub-problem

def Sub_problem_Generator(y, tx, ids, jet_nbr, bool_mass,ill_defined_feature_tensor):
    sub_output_array, Initial_sub_array, ids_sub = Sub_array_Generator(y, tx, ids, jet_nbr, bool_mass)    
    Initial_sub_array = Feature_remover(Initial_sub_array,jet_nbr, bool_mass, ill_defined_feature_tensor)
    return sub_output_array, Initial_sub_array, ids_sub