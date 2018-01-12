import numpy as np
import matplotlib.pyplot as plt
from sampler import *

# Parameter setting
N = 10000 # experiment times

# Categorical Example
ap = [0.2,0.3,0.4,0.1]
bins = [0,1,2,3,4]
C = Categorical(ap)
sampleList = []
for i in xrange(N):
	sampleList.append(C.sample())
plt.hist(sampleList,bins,density = True, align = 'right')
plt.xlabel('Categories')
plt.ylabel('Probability')
plt.title('Histogram of Categorical Distribution(4)')
plt.grid(True)
plt.show()

# Univariate normal distribution Example
mu = 10
sigma = 1
G = UnivariateNormal(mu,sigma)
sampleList = []
for i in xrange(N):
	sampleList.append(G.sample())
plt.hist(sampleList,50,density = True)
plt.xlabel('random variable X')
plt.ylabel('Probability')
plt.title('Histogram of Normal Distribution N(10,1)')
plt.grid(True)
plt.show()

# MultiVariate Normal Distribution Example
Mu = np.array([1,1])
Sigma = np.array([[1,0.5],[0.5,1]])
MultiG = MultiVariateNormal(Mu,Sigma)
xList = []; yList = []
for i in xrange(N):
	sample = MultiG.sample()
	xList.append(sample[0])
	yList.append(sample[1])
plt.scatter(xList, yList,s = plt.rcParams['lines.markersize'] ** 2 * 0.25)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter of 2D Normal Distribution N([1,1],[[1,0.5],[0.5,1]])')
plt.show()

# Mixture Distribution Example
pm = []
Mu = np.array([1,1])
Sigma = np.array([[1,0],[0,1]])
pm.append(MultiVariateNormal(Mu,Sigma))
Mu = np.array([-1,1])
Sigma = np.array([[1,0],[0,1]])
pm.append(MultiVariateNormal(Mu,Sigma))
Mu = np.array([1,-1])
Sigma = np.array([[1,0],[0,1]])
pm.append(MultiVariateNormal(Mu,Sigma))
Mu = np.array([-1,-1])
Sigma = np.array([[1,0],[0,1]])
pm.append(MultiVariateNormal(Mu,Sigma))

ap = np.array([0.25,0.25,0.25,0.25])

Mixture = MixtureModel(ap,pm)

count = 0
for i in xrange(N):
	sample = Mixture.sample()
	if((sample[0]-0.1)**2 + (sample[1]-0.2)**2 < 1):
		count = count + 1
print('The Probability that falls in unit circle with center at (0.1,0.2) is:')
print count*1.0/N