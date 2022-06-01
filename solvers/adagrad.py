from scipy.optimize import OptimizeResult
import numpy as np
import numpy.linalg as lina

class AdaGrad(object):
    FEV_PER_ITER = 1
    def __init__(self, func, func_p, x0, D, eta = 1.0, epsilon = 1e-8):
        self.func = func
        self.func_p=func_p
        self.x= np.array(x0)
        self.x[:]= x0
        self.d = self.x.shape[0]
        self.D=D
        self.lam=np.zeros(shape=x0.shape)
        self.lam+=epsilon
        self.t=0.0
        self.eta = eta
        
    def update(self):
        self.t=self.t+1.0
        g=self.func_p(self.x)
        self.step(g)

        return self.x
        
       
    def evaluate(self, x):
        return self.func(x)
                

    def step(self,g):
        self.update_parameters(g)
        self.sgd(g)

    def update_parameters(self,g):
        self.lam += g**2
       
        
    def sgd(self,g):
        h=np.sqrt(self.lam)
        y = self.x - g/h
        y_norm=(lina.norm(y.flatten(),ord=1))
        if y_norm>self.D:
            a=1.0/np.sqrt(h)
            v=np.abs(y/a)
            x_sign=np.sign(y)
            x_val=self.project(v,a,self.D)*a
            self.x[:]=x_val*x_sign
        else:
            self.x[:]=y
    
    def project(self, v, a, c):
        mu_idx=np.argsort(v/a)[::-1]
        p1=np.cumsum(np.take_along_axis(a*v, mu_idx, axis=0))
        p2=np.cumsum(np.take_along_axis(a**2, mu_idx, axis=0))
        p3=np.take_along_axis(v/a, mu_idx, axis=0)
        rho = np.max(np.nonzero(p1-p2*p3<c))
        theta=(p1[rho]-c)/p2[rho]	
        return np.maximum(v-theta*a,0.0)
  
        



