import numpy as np

def pca(X):
    """
    Run PCA on dataset X
    U,S,V = pca(X) computes eigenvectors of the covariance matrix of X
    Return eigenvectors U and the eigenvalues in S
    """
    #########################################################
    #         YOUR CODE HERE                                #
    #########################################################
    
    # compute the covariance of X and then use the
    # svd function to compute the eigenvectors and
    # eigenvalues of the covariance matrix

    # When computing the covariance remember to divide by
    # the number of rows in X
    
    ########################################################
    #           END YOUR CODE                              #
    ########################################################
    return U,S,V

def feature_normalize(X):
    Xnorm = (X - X.mean(axis=0))/X.std(axis=0)
    return Xnorm, X.mean(axis=0), X.std(axis=0)

def project_data(X,U,K):
    """
    project_data computes the reduced data representation when projecting only 
    on to the top k eigenvectors
    Z = project_data(X, U, K) computes the projection of 
    the normalized inputs X into the reduced dimensional space spanned by
    the first K columns of U. It returns the projected examples in Z.
    """
    #########################################################
    #         YOUR CODE HERE                                #
    #########################################################
    
    
    
    ########################################################
    #           END YOUR CODE                              #
    ########################################################
    return Z


def recover_data(Z,U,K):
    """
    recover_data recovers an approximation of the original data when using the 
    projected principal axis U
    X_rec = recover_data(Z, U, K) recovers an approximation the 
    original data Z that has been reduced to K dimensions. It returns the
    approximate reconstruction in X_rec.
    """

    #########################################################
    #         YOUR CODE HERE                                #
    #########################################################
    
   

    ########################################################
    #           END YOUR CODE                              #
    ########################################################
    return X_rec
   
    
