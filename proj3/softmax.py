import numpy as np
from random import shuffle
import scipy.sparse


def softmax_loss_naive(theta, X, y, reg):
    """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - theta: d x K parameter matrix. Each column is a coefficient vector for class k
  - X: m x d array of data. Data are d-dimensional rows.
  - y: 1-dimensional array of length m with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to parameter matrix theta, an array of same size as theta
  """
  # Initialize the loss and gradient to zero.

    J = 0.0
    grad = np.zeros_like(theta)
    m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in J and the gradient in grad. If you are not              #
  # careful here, it is easy to run into numeric instability. Don't forget    #
  # the regularization term!                                                  #
  #############################################################################
    for i in xrange(m):
        h  = np.dot(X[i,:],theta)
        h_max = h.max()
        h_new = np.exp(h-h_max)
        p_all = np.sum(h_new)
        for j in xrange(theta.shape[1]):
            grad[:,j] = grad[:,j]- X[i,:]*(1*(y[i]==j) - h_new[j]/p_all)
        J= J - np.log(h_new[y[i]]/ float(p_all)) 
    J = J/float(m) + 0.5*reg*np.sum(theta**2)/float(m)
    
    grad = grad/float(m) + reg * theta/float(m)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

    return J, grad

  
def softmax_loss_vectorized(theta, X, y, reg):
    """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
    """
  # Initialize the loss and gradient to zero.

    J = 0.0
    grad = np.zeros_like(theta)
    m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in J and the gradient in grad. If you are not careful      #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization term!                                                      #
  #############################################################################
    
    h = np.dot(X, theta)
    h = h - h.max(axis=1).reshape((-1,1))
    ex = np.exp(h)
    s = ex.sum(axis = 1).reshape((-1,1))
    I = np.zeros(ex.shape)+np.arange(10);
    I = (I == y.reshape((-1,1)))*1
    J = np.log((ex/s)**I).sum()/(-m)+reg*np.sum(theta**2)/(2*m)
    grad = np.dot(X.T,I-ex/s)/(-m) + reg*theta/m    
                       
    
    
    
                   

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

    return J, grad
