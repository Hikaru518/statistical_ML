import numpy as np

class LinearClassifier(object):
    
    def __init__(self):
        self.theta = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,batch_size=200, verbose=False):
        """
        Train this linear classifier using gradient descent.

        Inputs:
        - X: A numpy array of shape (m, d) containing training data; there are m
          training samples each of dimension d.
        - y: A numpy array of shape (m,) containing training labels; y[i] = -1,1 for two class problems
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step. 
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        m,d = X.shape
        K  = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes

        if self.theta is None:
        # lazily initialize theta
            if K == 2:
                self.theta = 0.001 * np.random.randn((d,))
            else:
                self.theta = 0.001 * np.random.randn(d, K)
        
        # Run gradient descent to optimize theta
        loss_history = []
        for it in xrange(num_iters):

          # draw batch_size examples from X
            if batch_size < X.shape[0]:
                rand_idx = np.random.choice(m, batch_size)
                X_batch = X[rand_idx,:]
                y_batch = y[rand_idx]
            else:
                X_batch = X
                y_batch = y

      # evaluate loss and gradient
        loss, grad = self.loss(X_batch, y_batch, reg)
        loss_history.append(loss)

        # perform parameter update
        self.theta = self.theta - learning_rate * grad

        if verbose and it % 100 == 0:
            print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

        return loss_history

    def predict(self, X):
        """
        Use the coefficients of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: m x d array of training data. Each row is a d-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length m, and each element is an integer giving the
          predicted class.
        """
        pass
  
    def loss(self, X, y, reg):
        """
        Compute the loss function and its derivative. 
        Subclasses will override this.

        Inputs:
        - X: A numpy array of shape (m,d)
        - y: A numpy array of shape (m,) containing labels for X.

        - reg: (float) penalty term.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.theta; an array of the same shape as theta
        """
        pass

from linear_svm import binary_svm_loss, svm_loss_vectorized

class LinearSVM_twoclass(LinearClassifier):
    """ A subclass that uses the two-class SVM loss function """

    def loss(self, X, y, C):
        return binary_svm_loss(self.theta, X, y, C)

    def predict(self,X):
        y_pred = np.dot(X,self.theta)
        y_pred[y_pred < 0] = -1
        y_pred[y_pred > 0] = 1
 
        return y_pred 

class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X, y, reg):
        return svm_loss_vectorized(self.theta, X, y, reg)

    def predict(self,X):
        y_pred = np.zeros(X.shape[0])

        ##################################################################
        #   TODO: SVM prediction function, use X and self.theta          #
        #   return y_pred with class associated with each row of X       #
        #   1-2 lines of code expected                                   #
        ##################################################################
        y_pred = np.dot(X,self.theta)
        y_pred = np.argmax(y_pred,axis = 1)
        ##################################################################
        #   END OF YOUR CODE                                             #
        ##################################################################

    
        return y_pred
    
