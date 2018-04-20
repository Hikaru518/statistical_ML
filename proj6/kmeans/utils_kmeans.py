import numpy as np
import matplotlib.pyplot as plt

def find_closest_centroids(X, centroids):
    """
    find_closest_centroids computes the centroid memberships for every example
    idx = find_closest_centroids(X, centroids) returns the closest centroids
    in idx for a dataset X where each row is a single example. idx = m x 1 
    vector of centroid assignments (i.e. each entry in range [0..K-1])
    """

    # Set K
    K = centroids.shape[0]

    # You need to return the following variable correctly.
    idx = np.zeros((X.shape[0],),dtype=int)

    ######################### YOUR CODE HERE ########################################
    # Instructions: Go over every example, find its closest centroid, and store     #
    #               the index in the array idx at the appropriate location.         #
    #               Concretely, idx[i] should contain the index of the centroid     #
    #               closest to example i. Hence, it should be a value in the        #
    #               range 0..K-1                                                    #
    ################################################################################
    
   
    for i in range(X.shape[0]):
        n = 0
        s = X[i,:] - centroids[0]
        c = np.sum(s**2)
        for j in range(len(centroids)):
            s=X[i,:] - centroids[j]
            c_j = np.sum(s**2)
            if c_j<c:
                c = c_j
                n = j
        idx[i] = n
    
    


    ################################################################################
    #             END OF YOUR CODE                                                 #
    ################################################################################
    return idx


def compute_centroids(X, idx, K):
    """
    compute_centroids returs the new centroids by computing the means of the 
    data points assigned to each centroid.
    centroids = compute_centroids(X, idx, K) returns the new centroids by 
    computing the means of the data points assigned to each centroid. It is
    given a dataset X where each row is a single data point, a vector
    idx of centroid assignments (i.e. each entry in range [0..K-1]) for each
    example, and K, the number of centroids. You should return a matrix
    centroids, where each row of centroids is the mean of the data points
    assigned to it.
    """

    # You need to return the following variables correctly.
    centroids = np.zeros((K, X.shape[1]))

    ########################= YOUR CODE HERE ######################################
    # Instructions: Go over every centroid and compute mean of all points that    #
    #               belong to it. Concretely, the row vector centroids[i,:]       #
    #               should contain the mean of the data points assigned to        #
    #               centroid i.                                                   #
    ###############################################################################
    for i in range(K):
        s = np.zeros((1,X.shape[1]))
        c = 0
        for j in range(X.shape[0]):
            if idx[j] == i:
                s += X[j,:]
                c += 1
        centroids[i,] = s/c
   

    ################################################################################
    #             END OF YOUR CODE                                                 #
    ################################################################################
    return centroids


def kmeans_init_centroids(X,K):
    """
    This function initializes K centroids that are to be used on the dataset X.
    returns K initial centroids in X
    """
    centroids = np.zeros((K,X.shape[1]))
    #
    #######################= YOUR CODE HERE ######################################
    #  Construct a random permutation of the examples and pick the first K items  #                                              
    ###############################################################################
    import random
    r = random.sample(range(X.shape[0]),K)
    centroids = np.array([X[i,:] for i in r])
                      
    
    
    ################################################################################
    #             END OF YOUR CODE                                                 #
    ################################################################################
    return centroids



import matplotlib.cm as cm

def run_kmeans(X, initial_centroids, max_iters, plot_progress=False):
    """
    run_kmeans runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    [centroids, idx] = run_kmeans(X, initial_centroids, max_iters, ...
    plot_progress) runs the K-Means algorithm on data matrix X, where each 
    row of X is a single example. It uses initial_centroids used as the
    initial centroids. max_iters specifies the total number of interactions 
    of K-Means to execute. plot_progress is a true/false flag that 
    indicates if the function should also plot its progress as the 
    learning happens. This is set to false by default. run_kmeans returns 
    centroids, a Kxd matrix of the computed centroids and idx, a m x 1 
    vector of centroid assignments (i.e. each entry in range [0..K-1])
    """
   
    # Plot the data if we are plotting progress
    if plot_progress:
        plt.figure()

    # Initialize values
    m, d  = X.shape;
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1))

    # Run K-Means
    for i in range(max_iters):
    
        # Output progress
        print 'K-Means iteration ', i, max_iters
    
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids);
    
        # Optionally, plot progress here
        if plot_progress:
            colors = cm.rainbow(np.linspace(0,1,K))
            plot_progress_kmeans(X, idx,range(K),colors,'','',centroids, previous_centroids, idx, K, i);
            previous_centroids = centroids;

    
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K);

    return centroids, idx


def plot_progress_kmeans(X,y,labels,colors,xlabel,ylabel,centroids,previous_centroids,idx,K,iter):
    
    plt.title('Iteration '+ str(iter))
    for i in range(len(labels)):
        Xl = X[np.where(y==labels[i])]
        plt.scatter(Xl[:,0],Xl[:,1],c=colors[i], s=40)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # plot the centroids
    for i in range(len(centroids)):
        plt.plot([previous_centroids[i,0], centroids[i,0]], [previous_centroids[i,1], centroids[i,1]], color='k', linestyle='-', linewidth=2)
        plt.plot(previous_centroids[i,0],previous_centroids[i,1],color = 'black',marker='x', markersize=3,mew=5)
    
# X is two dimensional data (x1, x2)
# y is a vector of labels from 0 to K-1
# colors are a list of K colors, and legend is a list of K legends

def plot_cluster_data(X,labels,y,colors,xlabel,ylabel,legend):
    fig = plt.figure()
    for i in range(len(labels)):
        Xl = X[np.where(y==labels[i])]
        plt.scatter(Xl[:,0],Xl[:,1],c=colors[i], s=40, label = legend[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="upper right")




