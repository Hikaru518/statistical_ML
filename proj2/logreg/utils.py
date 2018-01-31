import numpy as np
from sklearn import model_selection
import logistic_regressor as lr
from sklearn import linear_model
import scipy.io

######################################################################################
#   The sigmoid function                                                             #
#     Input: z: can be a scalar, vector or a matrix                                  #
#     Output: sigz: sigmoid of scalar, vector or a matrix                            #
#     TODO: 1 line of code expected                                                  #
######################################################################################

def sigmoid(z):
    sig = np.zeros(z.shape)
    # Your code here
    
    sig = 1.0/(1+np.exp(-z))
    
    # End your code

    return sig

######################################################################################
#   The log_features transform                                                       #
#     Input: X: a data matrix                                                        #
#     Output: a matrix with every element x replaced by 1 + log(x)                   #
#     TODO: 1 line of code expected                                                  #
######################################################################################

def log_features(X):
    logf = np.zeros(X.shape)
    
    # Your code here
    
    logf = np.log(1+X)
    
    # End your code
    return logf

######################################################################################
#   The std_features transform                                                       #
#     Input: X: a data matrix                                                        #
#     Output: a matrix with every column with zero mean, unit variance               #
######################################################################################

def std_features(X):
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

######################################################################################
#   The bin_features transform                                                       #
#     Input: X: a data matrix                                                        #
#     Output: a matrix with every element x replaced by 1 if x > 0 else 0            #
#     TODO: 1 line of code expected                                                  #
######################################################################################

def bin_features(X):
    tX = np.zeros(X.shape)
    # your code here

    tX = X>0
    
    # end your code
    return tX

######################################################################################
#   The select_lambda_crossval function                                              #
#     Inputs: X: a data matrix                                                       #
#             y: a vector of labels                                                  #
#             lambda_low, lambda_high,lambda_step: range of lambdas to sweep         #
#             penalty: 'l1' or 'l2'                                                  #
#     Output: best lambda selected by crossvalidation for input parameters           #
######################################################################################

# Select the best lambda for training set (X,y) by sweeping a range of
# lambda values from lambda_low to lambda_high in steps of lambda_step
# pick sklearn's LogisticRegression with the appropriate penalty (and solver)

# 20-25 lines of code expected

# For each lambda value, divide the data into 10 equal folds
# using sklearn's cross_validation KFold function.
# Then, repeat i = 1:10:
#  1. Retain fold i for testing, and train logistic model on the other 9 folds
#  with that lambda
#  2. Evaluate accuracy of model on left out fold i
# Accuracy associated with that lambda is the averaged accuracy over all 10
# folds.
# Do this for each lambda in the range and pick the best one
#

def select_lambda_crossval(X,y,lambda_low,lambda_high,lambda_step,penalty):

    best_lambda = lambda_low

    # Your code here
    # Implement the algorithm above.
    from sklearn.model_selection import KFold
        
    lambda_tmp = lambda_low; accuracy_best = 0.0; accuracy_tmp = 0.0
    while(lambda_tmp<=lambda_high):
        kf = KFold(n_splits=10)
        accuracyC = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]            
            XX_train = np.vstack([np.ones((X_train.shape[0],)),X_train.T]).T
            XX_test  = np.vstack([np.ones((X_test.shape[0],)) ,X_test.T]).T
                        
            # train and calculate theta
            if (penalty == "l2"):
                sk_logreg = linear_model.LogisticRegression(penalty = 'l2', C=1.0/lambda_tmp,solver='lbfgs',fit_intercept=False)
                sk_logreg.fit(XX_train,y_train)
            if (penalty == "l1"):
                sk_logreg = linear_model.LogisticRegression(penalty = 'l1', C=1.0/lambda_tmp,solver='liblinear',fit_intercept=False)
                sk_logreg.fit(XX_train,y_train)

            # predict
            predy = sk_logreg.predict(XX_test)
            
            # calculate accuracy
            accuracy = np.mean(y_test == predy)
            accuracyC = accuracyC + accuracy
            
        accuracy_tmp = accuracyC/10.0
        
        if (accuracy_tmp > accuracy_best):
            accuracy_best = accuracy_tmp
            best_lambda = lambda_tmp
            
        lambda_tmp = lambda_tmp + lambda_step
        
    # end your code

    return best_lambda


######################################################################################

def load_mat(fname):
  
    d = scipy.io.loadmat(fname)
    Xtrain = d['Xtrain']
    ytrain = d['ytrain']
    Xtest = d['Xtest']
    ytest = d['ytest']

    return Xtrain, ytrain, Xtest, ytest


def load_spam_data():
    data = scipy.io.loadmat('spamData.mat')
    Xtrain = data['Xtrain']
    ytrain1 = data['ytrain']
    Xtest = data['Xtest']
    ytest1 = data['ytest']

    # need to flatten the ytrain and ytest
    ytrain = np.array([x[0] for x in ytrain1])
    ytest = np.array([x[0] for x in ytest1])
    return Xtrain,Xtest,ytrain,ytest

    
