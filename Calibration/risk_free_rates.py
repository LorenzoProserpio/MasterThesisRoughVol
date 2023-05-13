import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

tenor = np.array([0.019178082,0.038356164,0.057534247,0.083333333,0.166666667,0.25,0.333333333,0.416666667,0.5,0.75,1,1.5,2,3,4,5,6,7,8,9,10,12,15,20,25,30,40,50])
dates = np.array(["23-01-23","24-01-23","25-01-23","26-01-23","27-01-23","30-01-23","06-02-23","13-02-23","21-02-23"])
data = pd.read_csv("ratesOIS.csv")

def r(x, index = 0):
    rates = np.array(data[dates[index]])/100
    cs = CubicSpline(tenor, rates)
    return cs(x)
