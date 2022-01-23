from scipy.special import lambertw
from scipy.optimize import OptimizeResult
import numpy as np
import numpy.linalg as lina
class AOExpGrad(object):
    def __init__(self, func, func_p, x0, lower, upper, l1=1.0, l2=1.0,eta=1.0):
        self.func = func
        self.func_p=func_p
        self.x= np.zeros(shape=x0.shape)
        self.x[:]= x0
        self.d = self.x.size
        self.lower=lower
        self.upper=upper
        self.eta=eta
        self.lam=0.0
        self.l1=l1
        self.l2=l2
        
    def update(self):
        g=self.func_p(self.x)
        self.step(g)
        return self.x
        
    def step(self,g):
        self.update_parameters(g)
        if self.lam!=0.0:
            self.md(g)

    def update_parameters(self,g):
        self.lam+=((lina.norm(g.flatten(),ord=np.inf))**2)
        #self.lam+=g**2
       
        
    def md(self,g):
        beta = 1.0 / self.d
        alpha = np.sqrt(self.lam)/np.sqrt(np.log(self.d))*self.eta
        z=(np.log(np.abs(self.x) / beta + 1.0)) * np.sign(self.x) - g/ alpha
        y_val = beta*np.exp(abs(z))-beta
        x_sgn = np.sign(z)
        if self.l2 == 0.0:
            x_val = beta * np.exp(np.maximum(np.log(y_val/beta+1.0) - self.l1 / alpha,0.0)) - beta
        else:
            a = beta
            b = self.l2 / alpha
            c = np.minimum(self.l1/ alpha - np.log(y_val/beta+1.0),0.0)
            x_val = lambertw(a * b * np.exp(a * b - c), k=0).real / b - a
        y = x_sgn * x_val
        self.x = np.clip(y, self.lower, self.upper)

        

def fmin(func, func_p,x0, upper,lower,l1=1.0,l2=1.0, maxfev=50,callback=None, epoch_size=10,eta=1.0 ):
    alg=AOExpGrad(func=func,func_p=func_p,x0=x0,upper=upper,lower=lower,l1=l1,l2=l2,eta=eta)
    nit=maxfev
    fev=1
    y=None
    while fev <= maxfev:
        y=alg.update()
        if callback is not None and fev%epoch_size==0:
                res=OptimizeResult(func=func(y), x=y, nit=fev,
                          nfev=fev, success=(y is not None))
                callback(res)
        fev+=1
    return OptimizeResult(func=func(y), x=y, nit=nit,
                          nfev=fev, success=(y is not None))

