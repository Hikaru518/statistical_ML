import numpy as np
import matplotlib.pyplot as plt

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

# plot point data and line fit
def plot_data_and_fit(X,XX,y,theta,xlabel,ylabel,symbol):
    plt.figure()
    plt.plot(X,y,symbol)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(X,np.dot(XX,theta),'g+')
    plt.show()

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

