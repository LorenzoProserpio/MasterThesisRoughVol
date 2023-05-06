
import numpy as np

z1 = 0.23934445541427635
z2 = 0.23559167522825925
z3 = 0.19271882494190468

# Compute the initial forward variance curve at a given time t
# using the Gompertz function with precomputed parameters
def g(t):
    return z1 * np.exp(-z2 * np.exp(-z3 * t))
    
def variance_curve(t):
    t = t * 12
    return g(t)**2 + 2*t*z1**2*z2*z3*np.exp(-2*z2*np.exp(-z3*t)-z3*t)
