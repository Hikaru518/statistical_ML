import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import PolynomialFeatures

def plot_data(X,y,xlabel,ylabel):
    fig = plt.figure()
    plt.plot(X,y,'bo')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_line(X,y,xlabel,ylabel):
    fig = plt.figure()
    plt.plot(X,y,'b-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def make_surface_plot(X,Y,Z,xlabel,ylabel):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z,cmap=cm.jet)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel);


def make_contour_plot(X,Y,Z,levels,xlabel,ylabel,theta):
    plt.figure()
    CS = plt.contour(X, Y, Z, levels = levels)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot([theta[0]],[theta[1]], marker='x',color='r',markersize=10)


def plot_learning_curve(error_train,error_val,reg):
  plt.figure()
  xvals = np.arange(2,len(error_train)+1)
  plt.plot(xvals,error_train[1:],'b-',xvals,error_val[1:],'g-')
  plt.title('Learning curve for linear regression with lambda = '+str(reg))
  plt.xlabel('Number of training examples')
  plt.ylabel('Training/Validation error')
  plt.legend(["Training error","Validation error"])

def plot_fit(X,y,minx, maxx, mu, sigma, theta, p, xlabel, ylabel, title):

    plt.figure()
    plt.plot(X,y,'bo')

    # plots a learned polynomial regression fit 

    x = np.arange(minx - 5,maxx+15,0.1)
    # map the X values
    poly = sklearn.preprocessing.PolynomialFeatures(p,include_bias=False)
    x_poly = poly.fit_transform(np.reshape(x,(len(x),1)))
    x_poly = (x_poly - mu) / sigma
    # add the column of ones
    xx_poly = np.vstack([np.ones((x_poly.shape[0],)),x_poly.T]).T
    plt.plot(x,np.dot(xx_poly,theta))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def plot_lambda_selection(reg_vec,error_train,error_val):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.plot(reg_vec,error_train,'b-',reg_vec,error_val,'g-')
  plt.title('Variation in training/validation error with lambda')
  plt.xlabel('Lambda')
  plt.ylabel('Training/Validation error')
  plt.legend(["Training error","Validation error"])
  ax.set_xscale('log')



