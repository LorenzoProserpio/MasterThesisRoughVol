import numpy as np
from scipy.signal import convolve
from numpy.random import default_rng
from utils_rBergomi import *
import risk_free_rates as rf
import implied_q as iq

# Class for generating paths of the rBergomi model.
class rBergomi(object):

    def __init__(self, n, N, T, a):

        # Basic assignments
        self.T = T                                                   # Maturity
        self.n = n                                                   # Steps per year
        self.dt = 1.0/self.n                                         # Step size
        self.s = np.round(self.n * self.T).astype(int)               # Number of total steps
        self.t = np.linspace(0, self.T, 1 + self.s)[np.newaxis,:]    # Time grid
        self.a = a                                                   # Alpha
        self.N = N                                                   # Number of paths

        # Construct hybrid scheme correlation structure with kappa = 1
        self.e = np.array([0,0])
        self.c = cov(self.a, self.n)

    def dW1(self):
        np.random.seed(0)
        # Produces random numbers for variance process with required covariance structure
        #rng = default_rng()
        #return rng.multivariate_normal(self.e, self.c, (self.N, self.s), method = 'cholesky')
        #return np.random.multivariate_normal(self.e, self.c, (self.N, self.s))
        N = int(self.N/2)
        w = np.random.multivariate_normal(self.e, self.c, (N, self.s))
        return np.concatenate((w,-w), axis = 0)
    
    def dW2(self):
        np.random.seed(0)
        #Obtain orthogonal increments
        #rng = default_rng(0)
        #return rng.standard_normal((self.N, self.s)) * np.sqrt(self.dt)
        #return np.random.randn(self.N, self.s) * np.sqrt(self.dt)
        N = int(self.N/2)
        w = np.random.randn(N, self.s) * np.sqrt(self.dt)
        return np.concatenate((w,-w), axis = 0)

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

    def dZ(self, dW1, dW2, rho):
        # Constructs correlated price Brownian increments, dB
        
        self.rho = rho
        return rho * dW1[:,:,0] + np.sqrt(1 - rho**2) * dW2

    def V(self, Y, xi, eta):
        # rBergomi variance process.
        self.xi = xi
        self.eta = eta
        a = self.a
        t = self.t
        return xi * np.exp(eta * Y - 0.5 * eta**2 * t**(2 * a + 1))
        #return xi * ne.evaluate('exp(eta * Y - 0.5 * eta**2 * t**(2 * a + 1))')

    def S_all_path(self, V, dZ, r, q, S0):
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
        
    def S(self, V, dZ, r, q, S0):
        # rBergomi price process.
        self.S0 = S0
        dt = self.dt
        rho = self.rho

        # Construct non-anticipative Riemann increments
        exponent = np.zeros(self.N)
        for i in range(self.s):
            exponent += np.sqrt(V[:,i]) * dZ[:,i] - 0.5 * V[:,i] * dt + (r - q) * dt
        
        return S0 * np.exp(exponent)
        
    def global_S(self, V, dZ, S0, steps):
        # rBergomi price process.
        self.S0 = S0
        dt = self.dt
        rho = self.rho
        
        r = np.zeros(self.s)
        r[:steps[0]-1] = rf.r(self.t[0,steps[0]])
        r[steps[0]-1:] = rf.r(self.t[0,steps[0]:])
        q = list()
        for i in range(self.s):
            if i <= steps[0]:
                q.append(iq.q(self.t[0,steps[0]]))
            else:
                q.append(iq.q(self.t[0,i]))
        q = np.array(q)
        
        S = list()
        logS = np.log(S0)
        for i in range(self.s):
            logS += np.sqrt(V[:,i]) * dZ[:,i] - 0.5 * V[:,i] * dt + (r[i] - q[i]) * dt
            if i in steps:
                S.append(np.exp(logS))
                
        S.append(np.exp(logS))
                
        return np.array(S)
        
    def S1(self, V, dW1, rho, S0, r, q):
        """
        rBergomi parallel price process.
        """
        dt = self.dt

        # Construct non-anticipative Riemann increments
        increments = rho * np.sqrt(V[:,:-1]) * dW1[:,:,0] - 0.5 * rho**2 * V[:,:-1] * dt + rho * (r - q) * dt

        # Cumsum is a little slower than Python loop.
        integral = np.cumsum(increments, axis = 1)

        S = np.zeros_like(V)
        S[:,0] = S0
        S[:,1:] = S0 * np.exp(integral)
        return S
