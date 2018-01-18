from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from reg_linear_regressor_multi import RegularizedLinearReg_SquaredLoss
import plot_utils


#############################################################################
#  Normalize features of data matrix X so that every column has zero        #
#  mean and unit variance                                                   #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     Output: mu: D x 1 (mean of X)                                         #
#          sigma: D x 1 (std dev of X)                                      #
#         X_norm: N x D (normalized X)                                      #
#############################################################################

def feature_normalize(X):

    ########################################################################
    # TODO: modify the three lines below to return the correct values
    mu = np.mean(X,axis = 0)
    sigma = np.std(X,axis = 0)
    X_norm = (X-mu)/sigma
  
    ########################################################################
    return X_norm, mu, sigma


#############################################################################
#  Plot the learning curve for training data (X,y) and validation set       #
# (Xval,yval) and regularization lambda reg.                                #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: N x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#     reg: regularization strength (float)                                  #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def learning_curve(X,y,Xval,yval,reg):
    num_examples,dim = X.shape
    error_train = np.zeros((num_examples,))
    error_val = np.zeros((num_examples,))
    
    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 7 lines of code expected                                                #
    ###########################################################################
  
    reglinear_reg_lc = RegularizedLinearReg_SquaredLoss()
    for i in xrange(num_examples):
        X_i = X[:i+1]; y_i = y[:i+1]
        #theta_i = reglinear_reg_lc.train(X_i,y_i,reg=0.0,num_iters=1000)
        theta_i = reglinear_reg_lc.train(X_i,y_i,reg,num_iters=1000)
        error_train[i] = reglinear_reg_lc.loss(theta_i,X_i,y_i,0)
        error_val[i] = reglinear_reg_lc.loss(theta_i,Xval,yval,0)
    

    ###########################################################################

    return error_train, error_val

#############################################################################
#  Plot the validation curve for training data (X,y) and validation set     #
# (Xval,yval)                                                               #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#                                                                           #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def validation_curve(X,y,Xval,yval):
    reg_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 15,30, 50, 80, 100]
    error_train = np.zeros((len(reg_vec),))
    error_val = np.zeros((len(reg_vec),))

    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 5 lines of code expected                                                #
    ###########################################################################
    reglinear_reg_vc = RegularizedLinearReg_SquaredLoss()
    for i in xrange(len(reg_vec)):
        theta_i = reglinear_reg_vc.train(X,y,reg_vec[i],num_iters=1000)
        error_train[i] = reglinear_reg_vc.loss(theta_i,X,y,0)
        error_val[i] = reglinear_reg_vc.loss(theta_i,Xval,yval,0)
    
    return reg_vec, error_train, error_val

import random

#############################################################################
#  Plot the averaged learning curve for training data (X,y) and             #
#  validation set  (Xval,yval) and regularization lambda reg.               #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#     reg: regularization strength (float)                                  #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def averaged_learning_curve(X,y,Xval,yval,reg):
    num_examples,dim = X.shape
    error_train = np.zeros((num_examples,))
    error_val = np.zeros((num_examples,))

    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 10-12 lines of code expected                                            #
    ###########################################################################
    
    for i in xrange(num_examples):
        error_train_vec = []; error_val_vec = []
        for count in xrange(50):
            index = range(num_examples)
            np.random.shuffle(index)
            index = index[:i+1]
            X_i = X[index]; y_i = y[index]
            Xval_i = Xval[index]; yval_i = yval[index]
            reglinear_reg_alc = RegularizedLinearReg_SquaredLoss()
            theta_i = reglinear_reg_alc.train(X_i,y_i,reg,num_iters=1000)
            error_train_vec.append(reglinear_reg_alc.loss(theta_i,X_i,y_i,0))
            error_val_vec.append(reglinear_reg_alc.loss(theta_i,Xval_i,yval_i,0))
        error_train[i] = np.mean(error_train_vec)
        error_val[i] = np.mean(error_val_vec)

    ###########################################################################
    return error_train, error_val


#############################################################################
# Utility functions
#############################################################################
    
def load_mat(fname):
  d = scipy.io.loadmat(fname)
  X = d['X']
  y = d['y']
  Xval = d['Xval']
  yval = d['yval']
  Xtest = d['Xtest']
  ytest = d['ytest']

  # need reshaping!

  X = np.reshape(X,(len(X),))
  y = np.reshape(y,(len(y),))
  Xtest = np.reshape(Xtest,(len(Xtest),))
  ytest = np.reshape(ytest,(len(ytest),))
  Xval = np.reshape(Xval,(len(Xval),))
  yval = np.reshape(yval,(len(yval),))

  return X, y, Xtest, ytest, Xval, yval









