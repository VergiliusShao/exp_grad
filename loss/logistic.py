import numpy as np
def logistic_loss(w,x,y):
    return np.log(1.0+np.exp(-y*(np.dot(x,w))))
def logistic_grad(w,x,y):
    return -y*(1.0/(1.0+np.exp(y*np.dot(x,w))))*x
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def correntropy(w,x,y):
    return 50*(1.0-np.exp(-(y-np.dot(x,w))**2/100))