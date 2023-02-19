import numpy as np

# Compute the initial forward variance curve at a given time t
# using the Gompertz function with precomputed parameters
def variance_curve(t):
    z1 = 0.23934445541427635
    z2 = 0.23559167522825925
    z3 = 0.19271882494190468
    return z1 * np.exp(-z2 * np.exp(-z3 * t))