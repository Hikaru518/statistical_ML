import numpy as np
import scipy

class RegularizedLinearRegressor_Multi:

    def __init__(self):
        self.theta = None


    def train(self,X,y,reg=1e-5,num_iters=100):

        """
        Train a linear model using regularized  gradient descent.
        
        Inputs:
        - X: N X D array of training data. Each training point is a D-dimensional
         row.
        - y: 1-dimensional array of length N with values in the reals.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing


        Outputs:
        optimal value for theta
        """
    
        num_train,dim = X.shape
        theta = np.ones((dim,))

        # Run scipy's fmin algorithm to run the gradient descent
        theta_opt = scipy.optimize.fmin_bfgs(self.loss, theta, fprime = self.grad_loss, args=(X,y,reg),maxiter=num_iters)
            
        
        return theta_opt

    def loss(self, *args):
        """
        Compute the loss function and its derivative. 
        Subclasses will override this.

        Inputs: (in *args as a tuple)
        - theta: D+1 vector
        - X: N x D array of data; each row is a data point.
        - y: 1-dimensional array of length N with real values.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.theta; an array of the same shape as theta
        """
        
        pass

    def grad_loss(self, *args):
        """
        Compute the loss function and its derivative. 
        Subclasses will override this.

        Inputs: (in *args as a tuple)
        - theta: D+1 vector
        - X: N x D array of data; each row is a data point.
        - y: 1-dimensional array of length N with real values.
        - reg: (float) regularization strength.

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
        #  1 line of code expected                                                #
        ###########################################################################

        y_pred = np.dot(X, self.theta)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def normal_equation(self,X,y,reg):
        """
        Solve for self.theta using the normal equations.
        """
        ###########################################################################
        # TODO:                                                                   #
        # Solve for theta_n using the normal equation.                            #
        #  One line of code expected                                              #
        ###########################################################################

        theta_n = np.zeros((X.shape[1],))

        ###########################################################################
        return theta_n

class RegularizedLinearReg_SquaredLoss(RegularizedLinearRegressor_Multi):
    "A subclass of Linear Regressors that uses the squared error loss function """

    """
    Function that returns loss and gradient of loss with respect to (X, y) and
    self.theta
        - loss J is a single float
        - gradient with respect to self.theta is an array of the same shape as theta

    """

    def loss (self,*args):
        theta,X,y,reg = args

        num_examples,dim = X.shape
        J = 0
        grad = np.zeros((dim,))
        ###########################################################################
        # TODO:                                                                   #
        # Calculate J (loss) wrt to X,y, and theta.                               #
        #  2 lines of code expected                                               #
        ###########################################################################
        err = np.dot(X, theta) - y 
        J = 0.5*np.sum(np.square(err))/num_examples + 0.5*dim*reg*np.sum(np.square(theta[1:]))/num_examples

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return J

    def grad_loss(self,*args):                                                                          
        theta,X,y,reg = args
        num_examples,dim = X.shape
        grad = np.zeros((dim,))

        ###########################################################################
        # TODO:                                                                   #
        # Calculate gradient of loss function wrt to X,y, and theta.              #
        #  3 lines of code expected                                               #
        ###########################################################################
        
        grad[0] = np.sum((np.dot(X, theta)-y)*(X[:,0]))/num_examples
        grad[1:] = np.dot(np.dot(X, theta)-y,X[:,1:])/num_examples + reg*theta[1:]/num_examples

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return grad
