import numpy as np
import matplotlib.pyplot as plt
import logistic_regressor as lr
import sklearn
from sklearn import preprocessing
import utils

def plot_twoclass_data(X,y,xlabel,ylabel,legend):
    fig = plt.figure()
    X0 = X[np.where(y==0)]
    X1 = X[np.where(y==1)]
    plt.scatter(X0[:,0],X0[:,1],c='red', s=40, label = legend[0])
    plt.scatter(X1[:,0],X1[:,1],c='green', s = 40, label=legend[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="upper right")

def plot_decision_boundary(X,y,theta,  xlabel, ylabel, legend):
    plot_twoclass_data(X,y,xlabel,ylabel,legend)
    
    # create a mesh to plot in
    h = 0.01
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))

    # make predictions on this mesh
    lr1 = lr.LogisticRegressor()
    lr1.theta = theta
    Z = np.array(lr1.predict(np.c_[np.ones(xx1.ravel().shape),xx1.ravel(), xx2.ravel()]))

    # Put the result into a color contour plot
    Z = Z.reshape(xx1.shape)
#    plt.contourf(xx1, xx2, Z, cmap=plt.cm.Paired, alpha=0.5) # if you want a surface
    plt.contour(xx1,xx2,Z,cmap=plt.cm.gray,levels=[0.5])


def plot_decision_boundary_sklearn(X,y,sk_logreg,  xlabel, ylabel, legend):
    plot_twoclass_data(X,y,xlabel,ylabel,legend)
    
    # create a mesh to plot in
    h = 0.01
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))

    # make predictions on this mesh
    Z = np.array(sk_logreg.predict(np.c_[np.ones(xx1.ravel().shape),xx1.ravel(), xx2.ravel()]))

    # Put the result into a color contour plot
    Z = Z.reshape(xx1.shape)
#    plt.contourf(xx1, xx2, Z, cmap=plt.cm.Paired, alpha=0.5)
    plt.contour(xx1,xx2,Z,cmap=plt.cm.gray,levels=[0.5])


def plot_decision_boundary_poly(X,y,theta,reg,p,  xlabel, ylabel, legend):
    plot_twoclass_data(X,y,xlabel,ylabel,legend)
    
    # create a mesh to plot in
    h = 0.01
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))

    # make predictions on this mesh
    poly = sklearn.preprocessing.PolynomialFeatures(degree=p,include_bias=False)
    X_poly = poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()])
    XX = np.vstack([np.ones((X_poly.shape[0],)),X_poly.T]).T    
    
    lr1 = lr.LogisticRegressor()
    lr1.theta = theta
    Z = np.array(lr1.predict(XX))

    # Put the result into a color contour plot
    Z = Z.reshape(xx1.shape)
#    plt.contourf(xx1, xx2, Z, cmap=plt.cm.Paired, alpha=0.5)
    plt.contour(xx1,xx2,Z,cmap=plt.cm.gray,levels=[0.5])
    plt.title("Decision boundary for lambda = " + str(reg))

def plot_decision_boundary_sklearn_poly(X,y,sk_logreg,reg,p,  xlabel, ylabel, legend):
    plot_twoclass_data(X,y,xlabel,ylabel,legend)
    
    # create a mesh to plot in
    h = 0.01
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))

    # make predictions on this mesh
    poly = sklearn.preprocessing.PolynomialFeatures(degree=p,include_bias=False)
    X_poly = poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()])
    XX = np.vstack([np.ones((X_poly.shape[0],)),X_poly.T]).T    
    
    Z = np.array(sk_logreg.predict(XX))

    # Put the result into a color contour plot
    Z = Z.reshape(xx1.shape)
#    plt.contourf(xx1, xx2, Z, cmap=plt.cm.Paired, alpha=0.5)
    plt.contour(xx1,xx2,Z,cmap=plt.cm.gray,levels=[0.5])
    plt.title("Decision boundary for lambda = " + str(reg))



import sklearn
from sklearn import svm, linear_model
from sklearn.svm import l1_min_c

# From
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_path.html#example-linear-model-plot-logistic-path-py

def plot_regularization_path(X,y):
    plt.figure()
    cs = sklearn.svm.l1_min_c(X, y, loss='log') * np.logspace(0, 3)
    print("Computing regularization path ...")
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    coefs_ = []
    for c in cs:
        clf.set_params(C=c)
        clf.fit(X, y)
        coefs_.append(clf.coef_.ravel().copy())

    coefs_ = np.array(coefs_)
    plt.plot(np.log10(cs), coefs_)
    ymin, ymax = plt.ylim()
    plt.xlabel('log(C)')
    plt.ylabel('Coefficients')
    plt.title('Logistic Regression Path')

