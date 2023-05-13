import numpy as np
from scipy.stats import norm

def BSCall(S0, K, T, r, q, sigma):
    
    # Price of a call under Black&Scholes
    
    # S0: spot price
    # K: strike
    # T: years to expiration
    # r: risk free rate (1 = 100%)
    # q: annual yield
    # sigma: volatility (1 = 100%)
    

    sig = sigma*np.sqrt(T)
    d1 = (np.log(S0/K) + (r-q)*T)/sig + sig/2.
    d2 = d1 - sig
    
    return S0*norm.cdf(d1) - K*np.exp(-(r-q)*T)*norm.cdf(d2)

def BSPut(S0, K, T, r, q, sigma):
    
    # Price of a put under Black&Scholes
    
    # S0: spot price
    # K: strike
    # T: years to expiration
    # r: risk free rate (1 = 100%)
    # q: annual yield
    # sigma: volatility (1 = 100%)
    
    return BSCall(S0, K, T, r, q, sigma) + K*np.exp(-(r-q)*T) - S0

def BSImpliedVol(S0, K, T, r, q, P, Option_type = 1, toll = 1e-10):
    
    # Calculate implied volatility from prices using bisection
    
    # NOTE: All the parameters can be np.array(), except for P that MUST be a np.array().
    
    # S0: spot price
    # K: strike
    # T: years to expiration
    # r: risk free rate (1 = 100%)
    # q: annual yield
    # P: prices
    # Option_type: 1 for calls, 0 for puts
    # toll: error in norm 1 
    
    if Option_type:
        BSFormula = np.vectorize(BSCall)
    else:
        BSFormula = np.vectorize(BSPut)
    
    N = P.shape[0]
    sigma_low = 1e-10*np.ones(N)
    sigma_high = 10*np.ones(N)
    
    P_low = BSFormula(S0, K, T, r, q, sigma_low)
    P_high = BSFormula(S0, K, T, r, q, sigma_high)
    
    while np.sum(P_high - P_low) > toll:
        sigma = (sigma_low + sigma_high)/2.
        P_mean = BSFormula(S0, K, T, r, q, sigma)
        P_low += (P_mean < P)*(P_mean - P_low)
        sigma_low += (P_mean < P)*(sigma - sigma_low)
        P_high += (P_mean >= P)*(P_mean - P_high)
        sigma_high += (P_mean >= P)*(sigma - sigma_high)
        
    return sigma
