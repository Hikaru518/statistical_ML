import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

#### Implement the Gaussian kernel here ####

def gaussian_kernel(x1,x2,sigma):
    return np.exp( - float(np.sum((x1-x2)**2))/float(sigma**2)/2.0)

#### End of your code ####

# load a mat file

    
def load_mat(fname):
    data = scipy.io.loadmat(fname)
    X = data['X']
    y = data['y'].flatten()
    return X,y

def loadval_mat(fname):
    data = scipy.io.loadmat(fname)
    X = data['X']
    y = data['y'].flatten()
    X_val = data['Xval']
    y_val = data['yval'].flatten()
    return X,y, X_val, y_val

# plot training data

def plot_twoclass_data(X,y,xlabel,ylabel,legend):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_adjustable('box')
    X0 = X[np.where(y==0)]
    X1 = X[np.where(y==1)]
    plt.scatter(X0[:,0],X0[:,1],c='red', s=80, label = legend[0])
    plt.scatter(X1[:,0],X1[:,1],c='green', s = 80, label=legend[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="upper right")
    return ax

def plot_decision_boundary_sklearn(X,y,clf,  xlabel, ylabel, legend):

    plot_twoclass_data(X,y,xlabel,ylabel,legend)
    
    # create a mesh to plot in

    h = 0.01
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))

    # make predictions on this mesh
    Z = np.array(clf.predict(np.c_[xx1.ravel(), xx2.ravel()]))

    # Put the result into a color contour plot
    Z = Z.reshape(xx1.shape)
    plt.contour(xx1,xx2,Z,cmap=plt.cm.gray,levels=[0.5])


def plot_decision_boundary(X,y,clf,  xlabel, ylabel, legend):

    plot_twoclass_data(X,y,xlabel,ylabel,legend)
    
    # create a mesh to plot in

    h = 0.01
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))

    # make predictions on this mesh (but add intercept term)
    Z = np.array(clf.predict(np.c_[np.ones((xx1.ravel().shape[0],)), xx1.ravel(), xx2.ravel()]))

    # Put the result into a color contour plot
    Z = Z.reshape(xx1.shape)
    plt.contour(xx1,xx2,Z,cmap=plt.cm.gray,levels=[0.5])




def plot_decision_kernel_boundary(X,y,scaler, sigma, clf,  xlabel, ylabel, legend):

    ax = plot_twoclass_data(X,y,xlabel,ylabel,legend)
    ax.autoscale(False)

    # create a mesh to plot in

    h = 0.05
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))

    ZZ = np.array(np.c_[xx1.ravel(), xx2.ravel()])
    K = np.array([gaussian_kernel(x1,x2,sigma) for x1 in ZZ for x2 in X]).reshape((ZZ.shape[0],X.shape[0]))

    # need to scale it
    scaleK = scaler.transform(K)

    KK = np.vstack([np.ones((scaleK.shape[0],)),scaleK.T]).T

    # make predictions on this mesh (but add intercept term)
    Z = clf.predict(KK)

    # Put the result into a color contour plot
    Z = Z.reshape(xx1.shape)
    plt.contour(xx1,xx2,Z,cmap=plt.cm.gray,levels=[0.5])


def get_vocab_dict():
    words = {}
    inv_words = {}
    f = open('data/vocab.txt','r')
    for line in f:
        if line != '':
            (ind,word) = line.split('\t')
            words[int(ind)] = word.rstrip('\n')
            inv_words[word.rstrip('\n')] = int(ind)
    return words, inv_words
