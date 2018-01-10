import numpy as np
import matplotlib.pyplot as plt

class ProbabilityModel:

    # Returns a single sample (independent of values returned on previous calls).
    # The returned value is an element of the model's sample space.
    def sample(self):
        pass


# The sample space of this probability model is the set of real numbers, and
# the probability measure is defined by the density function 
# p(x) = 1/(sigma * (2*pi)^(1/2)) * exp(-(x-mu)^2/2*sigma^2)
class UnivariateNormal(ProbabilityModel):
    
    # Initializes a univariate normal probability model object
    # parameterized by mu and (a positive) sigma
    def __init__(self,mu,sigma):
        self.Mu=Mu
        self.Sigma=Sigma

    def sample(self):
    	sampleRes = 0
    	Num_UnifromD = 25
    	# Use the sum of 25 i.i.d uniform distribution to approximate the normal distribution 
    	for i in xrange(Num_UnifromD): 
    		randNum = np.random.uniform(-np.sqrt(3),np.sqrt(3))
    		sampleRes = sampleRes + randNum
    	sampleRes = np.sqrt(Num_UnifromD)*sampleRes
    	sampleRes = sampleRes*self.sigma + self.Mu
    	return sampleRes


# The sample space of this probability model is the set of D dimensional real
# column vectors (modeled as numpy.array of size D x 1), and the probability 
# measure is defined by the density function 
# p(x) = 1/(det(Sigma)^(1/2) * (2*pi)^(D/2)) * exp( -(1/2) * (x-mu)^T * Sigma^-1 * (x-mu) )
class MultiVariateNormal(ProbabilityModel):
    
    # Initializes a multivariate normal probability model object 
    # parameterized by Mu (numpy.array of size D x 1) expectation vector 
    # and symmetric positive definite covariance Sigma (numpy.array of size D x D)
    def __init__(self,Mu,Sigma):
        self.Mu=Mu
        self.Sigma=Sigma

    def sample(self):
    	pass



# The sample space of this probability model is the finite discrete set {0..k-1}, and 
# the probability measure is defined by the atomic probabilities 
# P(i) = ap[i]
class Categorical(ProbabilityModel):
    
    # Initializes a categorical (a.k.a. multinom, multinoulli, finite discrete) 
    # probability model object with distribution parameterized by the atomic probabilities vector
    # ap (numpy.array of size k).
    def __init__(self,ap):
        self.ap=ap

    def sample(self):
    	N = len(self.ap)
    	Cumulative_ap = [sum(self.ap[:i]) for i in range(1,N+1)]

    	randNum = np.random.uniform(0,1)
    	for i in xrange(N):
    		if (randNum < Cumulative_ap[i]):
    			return i
    
# The sample space of this probability model is the union of the sample spaces of 
# the underlying probability models, and the probability measure is defined by 
# the atomic probability vector and the densities of the supplied probability models
# p(x) = sum ad[i] p_i(x)
class MixtureModel(ProbabilityModel):
    
    # Initializes a mixture-model object parameterized by the
    # atomic probabilities vector ap (numpy.array of size k) and by the tuple of 
    # probability models pm
    def __init__(self,ap,pm):
        pass

