import numpy as np

# TBSS kernel applicable to the rBergomi variance process.
def g(x, a):
    return x**a

# Optimal discretisation of TBSS process for minimising hybrid scheme error.
def b(k, a):
    return ((k**(a+1)-(k-1)**(a+1))/(a+1))**(1/a)

# Covariance matrix for given alpha and n, assuming kappa = 1.
def cov(a, n):
    cov = np.array([[0.,0.],[0.,0.]])
    cov[0,0] = 1./n
    cov[0,1] = 1./((1.*a+1) * n**(1.*a+1))
    cov[1,1] = 1./((2.*a+1) * n**(2.*a+1))
    cov[1,0] = cov[0,1]
    return cov