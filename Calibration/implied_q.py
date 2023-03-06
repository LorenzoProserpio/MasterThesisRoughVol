import numpy as np
import ImpliedDrift as iD
import risk_free_rates as rf

def q(tenor):
	return rf.r(tenor) - iD.implied_drift(tenor)