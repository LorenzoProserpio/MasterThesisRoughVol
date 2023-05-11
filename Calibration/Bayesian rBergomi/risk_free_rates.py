import numpy as np

tenor = np.array([0.019178082,0.038356164,0.057534247,0.083333333,0.166666667,0.25,0.333333333,0.416666667,0.5,0.75,1,1.5,2,3,4,5,6,7,8,9,10,12,15,20,25,30,40,50])
rates = np.array([4.3316,4.445,4.497,4.536,4.5875,4.668,4.7342,4.789,4.8269,4.878,4.8485,4.53825,4.201475,3.74869,3.511879,3.377621,3.297157,3.245,3.211946,3.195546,3.188184,3.189691,3.19873,3.163828,3.063784,2.96249,2.746381,2.532429])

def r(x):
	return np.interp(x, tenor, rates)/100