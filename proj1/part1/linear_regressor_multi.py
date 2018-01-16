import numpy as np

class LinearRegressor_Multi:

    def __init__(self):
        self.theta = None


    def train(self,X,y,learning_rate=1e-3, num_iters=100,verbose=False):

        """
        Train a linear model using gradient descent.
        
        Inputs:
        - X: N X D array of training data. Each training point is a D-dimensional
         row.
        - y: 1-dimensional array of length N with values in the reals.
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
    
        num_train,dim = X.shape
        if self.theta is None:
            # lazily initialize theta 
            self.theta = np.zeros((dim,))

        # Run gradient descent to find theta
        J_history = []
        for i in xrange(num_iters):
            # evaluate loss and gradient
            loss, grad = self.loss(X, y)
            J_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the parameters using the gradient and the learning rate.       #
            #   One line of code expected
            #########################################################################
            self.theta -= learning_rate*grad

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and i % 100 == 0:
                print 'iteration %d / %d: loss %f' % (i, num_iters, loss)

        return J_history

    def loss(self, X, y):
        """
        Compute the loss function and its derivative. 
        Subclasses will override this.

        Inputs:
        - X: N x D array of data; each row is a data point.
        - y: 1-dimensional array of length N with real values.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.theta; an array of the same shape as theta
        """
        pass

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: N x D array of training data. Each row is a D-dimensional point.

        Returns:
        - y_pred: Predicted output for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is a real number.
        """
        y_pred = np.zeros(X.shape[0])

        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted outputs in y_pred.           #
        #  One line of code expected                                              #
        ###########################################################################
        y_pred = np.dot(X, self.theta)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def normal_equation(self,X,y):
        """
        Solve for self.theta using the normal equations.
        """

        ###########################################################################
        # TODO:                                                                   #
        # Solve for theta_n using the normal equation.                            #
        #  One line of code expected                                              #
        ###########################################################################

        theta_n = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(),X)),X.transpose()),y)

        ###########################################################################

        return theta_n

class LinearReg_SquaredLoss(LinearRegressor_Multi):
    "A subclass of Linear Regressors that uses the squared error loss function """

    """
    Function that returns loss and gradient of loss with respect to (X, y) and
    self.theta
        - loss J is a single float
        - gradient with respect to self.theta is an array of the same shape as theta

    """

    def loss (self,X,y):
        num_examples,dim = X.shape
        J = 0
        grad = np.zeros((dim,))
        ###########################################################################
        # TODO:                                                                   #
        # Calculate J (loss) and grad (gradient) wrt to X,y, and self.theta.      #
        #  2-3 lines of code expected                                             #
        ###########################################################################
        err = np.dot(X, self.theta)-y
        J = np.sum(0.5*np.square(err)/len(y))
        grad = np.dot(X.T, err)/len(y)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return J, grad
