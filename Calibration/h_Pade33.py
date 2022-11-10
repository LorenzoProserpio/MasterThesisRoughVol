import numpy as np
from scipy.special import gamma as gamma

def h_Pade33(H, rho, eta, a, x):
    
    # Compute the Pade approximant in x
    
    # H is the Hurst parameter(?)
    # rho is the Heston???
    # eta is the Heston???
    # a is the order of the approximant that is m/n where m is the order of the numerator and n of the denominator(?)
    # x is the point in which we calculate the approximation
    
    al = H + .5
    a_2 = a**2
    aa = np.sqrt(a * (a + 1j) - rho**2 * a_2)
    rm = -1j * rho * a - aa
    rp = -1j * rho * a + aa
    
    gamma_1_p = gamma(1 + al)
    gamma_2_p = gamma(1 + 2 * al)
    gamma_3_p = gamma(1 + 3*al)
    
    b1 = -a * (a + 1j) / (2 * gamma_1_p)
    b2 = (1 - a*1j) * a_2 * rho / (2 * gamma_2_p)
    b3 = gamma_2_p / gamma_3_p * (a_2 * (1j+a)**2 / (8*gamma_1_p**2)+(a+1j)*a**3*rho**2 / (2*gamma_2_p))
    
    b12 = b1 * b2
    b13 = b1 * b3
    
    b1_2 = b1 ** 2
    b2_2 = b2 ** 2
    
    gamma_1_m = gamma(1 - al)
    gamma_2_m = gamma(1 - 2 * al)
    
    g0 = rm
    g1 = -rm / (aa * gamma_1_m)
    g2 = rm / (aa**2 * gamma_2_m) * (1 + rm * gamma_2_m / (2 * aa * gamma_1_m**2))
    
    g01 = g0 * g1
    g02 = g0 * g2
    
    g0_2 = g0 ** 2
    g1_2 = g1 ** 2
  
    den = g0**3 + 2*b1*g0*g1 - b2*g1**2 + b1**2*g2 + b2*g0*g2

    p1 = b1
    p2 = (b1_2 * g0_2 + b2 * g0**3 + b1**3 * g1 + b12 * g01 - b2_2 * g1_2 + b13 * g1_2 + b2_2 * g02 - b13 * g02)/den
    q1 = (b1 * g0_2 + b1_2 * g1 - b2 * g01 + b3 * g1_2 - b12 * g2 - b3 * g02) / den
    q2 = (b1_2 * g0 + b2 * g0_2 - b12 * g1 - b3 * g01 + b2_2 * g2 - b13 * g2) / den
    q3 = (b1**3 + 2*b12 * g0 + b3 * g0_2 - b2_2 * g1 + b13 * g1 ) / den
    p3 = g0*q3

    y = x**al
    
    h_pade = (p1*y + p2*y**2 + p3*y**3) / (1 + q1*y + q2*y**2 + q3*y**3) #h_pade is the result of the function h.Pade33
  
    res = 0.5 * (h_pade - rm) * (h_pade - rp) #res is the result of the function d.h.Pade33
  
    return np.array([h_pade, res])