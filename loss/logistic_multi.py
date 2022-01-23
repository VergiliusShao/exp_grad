import numpy as np
def logistic_loss(w,x,y):
    return np.log(1.0+np.exp(-y*(np.sum(x*w,axis=0))))
def logistic_grad(w,x,y):
    deri=-y*(1.0/(1.0+np.exp(y*np.sum(x*w,axis=0))))
    grad=np.zeros(shape=w.shape)
    for i in range(deri.size):
        grad[:,i]=x[:,i]*deri[i]
    return grad
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

#w = np.array([1.0,2.0])
#x = np.array([-1.0,1.0])
#y = 1.0
#print(logistic_loss(w,x,y))
#print(logistic_grad(w,x,y))