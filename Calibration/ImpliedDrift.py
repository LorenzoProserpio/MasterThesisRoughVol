import numpy as np

def implied_drift(tenor):
    '''Calculate the drift term for different tenor (in years)'''
    
    a,b,c,d = (0.13150334, 1.67001571, 9.41005645, 0.02267608)
    T_max = 3629.999999835/365.
    
    if tenor > 30/365. and tenor < T_max:
        return (a*(tenor/T_max)**(b-1)*(1-tenor/T_max)**(c-1))+d
    if tenor >= T_max:
        return d
    if tenor <= 30/365. and tenor >= 13.9/365.:
        return np.interp(tenor*365,[14,30.],[0.03449854534679646,0.027609263139159457])
    print('Error: cannot accept tenors under 14 days.')
