import numpy as np
from scipy.signal import convolve
from numpy.random import default_rng
from utils_rBergomi import *

# Class for generating paths of the rBergomi model.
class rBergomi(object):

    def __init__(self, n = 100, N = 1000, T = 1.00, a = -0.4):

        # Basic assignments
        self.T = T                                                   # Maturity
        self.n = n                                                   # Steps per year
        self.dt = 1.0/self.n                                         # Step size
        self.s = int(self.n * self.T)                                # Number of total steps
        self.t = np.linspace(0, self.T, 1 + self.s)[np.newaxis,:]    # Time grid
        self.a = a                                                   # Alpha
        self.N = N                                                   # Number of paths

        # Construct hybrid scheme correlation structure with kappa = 1
        self.e = np.array([0,0])
        self.c = cov(self.a, self.n)

    def dW1(self):
        # Produces random numbers for variance process with required covariance structure
        #rng = default_rng()
        #return rng.multivariate_normal(self.e, self.c, (self.N, self.s), method = 'cholesky')
        return np.random.multivariate_normal(self.e, self.c, (self.N, self.s))
    
    def dW2(self):
        #Obtain orthogonal increments
        #rng = default_rng(0)
        #return rng.standard_normal((self.N, self.s)) * np.sqrt(self.dt)
        return np.random.randn(self.N, self.s) * np.sqrt(self.dt)

    def Y(self, dW):
        #Constructs Volterra process from appropriately correlated 2d Brownian increments
        
        Y1 = np.zeros((self.N, 1 + self.s)) # Exact integrals
        Y2 = np.zeros((self.N, 1 + self.s)) # Riemann sums

        Y1[:,1 : self.s+1] = dW[:, :self.s, 1]   # Assumes kappa = 1

        # Construct arrays for convolution
        G = np.zeros(1 + self.s) # Gamma
        for k in np.arange(2, 1 + self.s, 1):
            G[k] = g(b(k, self.a)/self.n, self.a)

        X = dW[:,:,0] # Xi

        # Compute convolution and extract relevant terms
        for i in range(self.N):
            Y2[i,:] = np.convolve(G, X[i,:])[:1+self.s]

        # Finally contruct and return full process
        return np.sqrt(2 * self.a + 1) * (Y1 + Y2)

    def dZ(self, dW1, dW2, rho = 0.0):
        # Constructs correlated price Brownian increments, dB
        
        self.rho = rho
        return rho * dW1[:,:,0] + np.sqrt(1 - rho**2) * dW2

    def V(self, Y, xi = 1.0, eta = 1.0):
        # rBergomi variance process.
        self.xi = xi
        self.eta = eta
        a = self.a
        t = self.t
        return xi * np.exp(eta * Y - 0.5 * eta**2 * t**(2 * a + 1))
        #return xi * ne.evaluate('exp(eta * Y - 0.5 * eta**2 * t**(2 * a + 1))')

    def S(self, V, dZ, r, q, S0 = 1):
        # rBergomi price process.
        self.S0 = S0
        dt = self.dt
        rho = self.rho

        # Construct non-anticipative Riemann increments
        increments = np.sqrt(V[:,:-1]) * dZ - 0.5 * V[:,:-1] * dt + (r - q) * dt

        # Cumsum is a little slower than Python loop.
        integral = np.cumsum(increments, axis = 1)

        S = np.zeros_like(V)
        S[:,0] = S0
        S[:,1:] = S0 * np.exp(integral)
        #S[:,1:] = S0 * ne.evaluate('exp(integral)')
        return S