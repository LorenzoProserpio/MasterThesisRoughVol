import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

dates = np.array(["23-01-23","24-01-23","25-01-23","26-01-23","27-01-23","30-01-23","06-02-23","13-02-23","21-02-23"])
data = pd.read_csv("ratesOIS.csv")
tenor = np.array(data.TENOR)
F = np.array(pd.read_csv("implied_forward.csv"))
spot = np.array(pd.read_csv("spot.csv").Spot)
TENOR = np.array(pd.read_csv("hist_spx.csv")["Exp Date"])


def r(x, index = 0):
    rates = np.array(data[dates[index]])/100
    cs = CubicSpline(tenor, rates)
	if(x>20):
            return cs(20)
    return cs(x)
    
def drift(x, index = 0):
    S0 = spot[index]
    d = -np.log(S0/F).flatten()/TENOR
	cs = CubicSpline(TENOR, d)
	if(x>20):
            return cs(20)
    return cs(x)
    
def q(x, index = 0):
    return r(x, index) - drift(x, index)
