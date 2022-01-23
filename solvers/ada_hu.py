from scipy.optimize import OptimizeResult
import numpy as np
import numpy.linalg as lina

class AdaHU(object):
    def __init__(self, func, func_p, x0, D ):
        self.func = func
        self.func_p=func_p
        self.x= np.zeros(shape=x0.shape)
        self.x[:]= x0
        self.d = self.x.size
        self.D=D   
        self.lam=0.0

    def update(self):
        g=self.func_p(self.x)
        self.step(g)
        return self.x

    def step(self,g):
        self.update_parameters(g)
        if self.lam>0:
            self.md(g)

    def update_parameters(self,g):
        self.lam+=(lina.norm(g.flatten(),ord=np.inf)**2)
       
        
    def md(self,g):
        alpha=np.sqrt(self.lam)/np.sqrt(np.log(self.d))
        beta=1.0/self.d
        y= beta*np.sinh(np.arcsinh(self.x/beta)-g/alpha)
        y_norm=(lina.norm(y.flatten(),ord=1))
        if y_norm>self.D:
            self.x[:]=y/y_norm*self.D
        else:
            self.x[:]=y



