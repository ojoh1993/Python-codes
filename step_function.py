import numpy as np
import matplotlib.pylab as plt
def step_function(x):
	y = x>0
	return y.astype(np.int)
