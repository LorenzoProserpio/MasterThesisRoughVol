import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

dates = np.array(["23-01-23","24-01-23","25-01-23","26-01-23","27-01-23","30-01-23","06-02-23","13-02-23","21-02-23"])
data = pd.read_csv("ratesOIS.csv")
tenor = np.array(data.TENOR)
Forw = pd.read_csv("forw.csv")
spot = np.array(pd.read_csv("spot.csv").Spot)
TENOR = pd.read_csv("tenor.csv")


def r(x, index = 0):
    rates = np.array(data[dates[index]])/100
    cs = CubicSpline(tenor, rates)
    return cs(x)
    
def drift(x, index = 0):
    S0 = spot[index]
    F = np.array(Forw[dates[index]]).flatten()
    Tenor = np.array(TENOR[dates[index]]).flatten()
    d = -np.log(S0/F).flatten()/Tenor
    cs = CubicSpline(Tenor, d)
    return cs(x)
    
def q(x, index = 0):
    return r(x, index) - drift(x, index)
