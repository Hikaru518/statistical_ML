import numpy as np

##################################################################################
#   Two class or binary SVM                                                      #
##################################################################################

def binary_svm_loss(theta, X, y, C):
    """
  SVM hinge loss function for two class problem

  Inputs:
  - theta: A numpy vector of size d containing coefficients.
  - X: A numpy array of shape mxd 
  - y: A numpy array of shape (m,) containing training labels; +1, -1
  - C: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to theta; an array of same shape as theta
    """

    m, d = X.shape
    grad = np.zeros(theta.shape)
    J = 0

  ############################################################################
  # TODO                                                                     #
  # Implement the binary SVM hinge loss function here                        #
  # 4 - 5 lines of vectorized code expected                                  #
  ############################################################################
    diff = 1 - np.multiply(y,np.dot(X, theta))
    J = np.sum(theta**2)*0.5/m + C * np.sum(np.maximum( 0 , diff))/m
    indicator = np.multiply(y,np.dot(X,theta)) <1
    indicator = indicator
    X_new  = X[indicator,:]
    y_new = y[indicator]
    reg = np.dot(X_new.T, y_new)
    grad = theta/m - C * reg /m
    
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

    return J, grad

##################################################################################
#   Multiclass SVM                                                               #
##################################################################################

# SVM multiclass

def svm_loss_naive(theta, X, y, reg):
    """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension d, there are K classes, and we operate on minibatches
  of m examples.

  Inputs:
  - theta: A numpy array of shape d X K containing parameters.
  - X: A numpy array of shape m X d containing a minibatch of data.
  - y: A numpy array of shape (m,) containing training labels; y[i] = k means
    that X[i] has label k, where 0 <= k < K.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss J as single float
  - gradient with respect to weights theta; an array of same shape as theta
    """

    K = theta.shape[1] # number of classes
    m = X.shape[0]     # number of examples

    J = 0.0
    dtheta = np.zeros(theta.shape) # initialize the gradient as zero
    delta = 1.0

  #############################################################################
  # TODO:                                                                     #
  # Compute the loss function and store it in J.                              #
  # Do not forget the regularization term!                                    #
  # code above to compute the gradient.                                       #
  # 8-10 lines of code expected                                               #
  #############################################################################
    
    L = []
    for i in range(m):
        tempSum = 0
        tmp = 0
        for j in range(K):
            if(j != y[i]):
                tmp = max(0,np.dot(theta[:,j].T,X[i]) - np.dot(theta[:,y[i]].T,X[i]) + delta)
                tempSum += tmp
                if(tmp>0):
                    dtheta[:,j] += X[i]
                    dtheta[:,y[i]] -= X[i]
        L.append(tempSum)
        
    J = np.mean(L) + 0.5*reg*np.linalg.norm(theta)*np.linalg.norm(theta)
    
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dtheta.            #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

    dtheta = 1.0*dtheta/m + reg*theta
                       
    return J, dtheta

def svm_loss_vectorized(theta, X, y, reg):
    """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
    """

    J = 0.0
    dtheta = np.zeros(theta.shape) # initialize the gradient as zero
    delta = 1.0

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in variable J.                                                     #
  # 8-10 lines of code                                                        #
  #############################################################################
  
    m = X.shape[0] 

    scores = np.dot(X,theta)
    correct_class_scores = scores[np.arange(m), y]   # 1 by N
    correct_class_scores = np.reshape(correct_class_scores, (m, -1))  # N by 1
    margin=scores-correct_class_scores + delta 
    margin[np.arange(m),y]=0.0 
    margin[margin<=0]=0.0
    J+=np.sum(margin)
    J = J/m + 0.5*reg * np.sum(theta * theta) 
    


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dtheta.                                       #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
    
    margin[margin>0]=1.0
    row_sum = np.sum(margin,axis=1)
    margin[np.arange(m),y] = -row_sum
    dtheta = 1.0/m*np.dot(X.T,margin) + reg*theta;
    
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

    return J, dtheta
