from numpy.core.fromnumeric import size
import numpy as np
def logistic_loss(w,x,y):
    return np.log(1.0+np.exp(-y*(np.dot(x,w))))
def logistic_grad(w,x,y):
    return -y*(1.0/(1.0+np.exp(y*np.dot(x,w))))*x
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

#w = np.array([1.0,2.0])
#x = np.array([-1.0,1.0])
#y = 1.0
#print(logistic_loss(w,x,y))
#print(logistic_grad(w,x,y))