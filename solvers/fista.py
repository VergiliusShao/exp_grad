from scipy.optimize import OptimizeResult
import numpy as np
import numpy.linalg as lina
class FISTA(object):
    MAX_BACK_ITER=10
    def __init__(self, func, func_p, x0, lower, upper, l1=1.0, l2=1.0,eta=1.0):
        self.L=1.0
        self.func = func
        self.func_p=func_p
        self.x= np.zeros(shape=x0.shape)
        self.x[:]= x0
        self.y= np.zeros(shape=x0.shape)
        self.y[:]= x0
        self.d = self.x.size
        self.lower=lower
        self.upper=upper
        self.eta=np.e
        self.t=0.0
        self.l1=l1
        self.l2=l2
    

    def update(self):
        g=self.func_p(self.y)
        self.step(g)
        return self.y
                

    def step(self,g):
        x=self.update_parameters(g)
        self.t=(1+np.sqrt(1+4*self.t**2))/2
        self.y=x+(self.t-1)/self.t*(x-self.x)
        x=self.x

    def update_parameters(self,g):
        for i in range(0,self.MAX_BACK_ITER+1):
            _L=(self.eta**i)*self.L
            y=self.md(g,_L)
            if self.func(y)+self.G(y)<=self.func(self.y)+np.sum((y-self.y)*g)+0.5*_L*(np.linalg.norm((y-self.y).flatten(),ord=2)**2)+self.G(self.y):
                x=y
                break
        x=y
        self.L=_L
        return x    
    
    def G(self, y):
        return self.l1*np.linalg.norm(y.flatten(),ord=1)+0.5*self.l2*(np.linalg.norm(y.flatten(),ord=2)**2)
        
    def md(self,g,L):
        z = self.x-g/L
        x_sgn = np.sign(z)
        x_val = np.maximum(L*np.abs(z)-self.l1,0.0)/(L+self.l2)
        y = x_sgn * x_val
        return np.clip(y, self.lower, self.upper)

def fmin(func, func_p,x0, upper,lower,l1=1.0,l2=1.0, maxfev=50,callback=None, epoch_size=10,eta=1.0 ):
    alg=FISTA(func=func,func_p=func_p,x0=x0,upper=upper,lower=lower,l1=l1,l2=l2,eta=eta)
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

