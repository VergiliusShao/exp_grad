from scipy.optimize import OptimizeResult
import numpy as np
import numpy.linalg as lina
import math
def hinge_loss(w,x,y):
    return np.maximum(0.0,1.0-y*(np.dot(x,w)))
def hinge_grad(w,x,y):
    t=np.dot(x,w)
    g=np.zeros(x.shape)
    if y*t <1.0:
        g[:]=-y*x    
    return g