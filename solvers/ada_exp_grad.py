from scipy.optimize import OptimizeResult
import numpy as np
import numpy.linalg as lina
class AdaExpGrad(object):
    def __init__(self, func, func_p, x0, D, eta=1.0):
        self.func = func
        self.func_p=func_p
        self.x= np.array(x0)
        self.x[:]= x0
        self.d = self.x.size
        self.D=D
        self.lam=0.0
        self.t=0.0
        
    def update(self):
        self.t=self.t+1.0
        g=self.func_p(self.x)
        self.step(g)
        return self.x
        
       
    def evaluate(self, x):
        return self.func(x)
                

    def step(self,g):
        self.update_parameters(g)
        if self.lam >0:
            self.md(g)

    def update_parameters(self,g):
        self.lam+=(lina.norm(g.flatten(),ord=np.inf)**2)
        #self.lam+=g**2
       
        
    def md(self,g):
        beta=1.0/self.d
        alpha=np.sqrt(self.lam)/np.sqrt(np.log(self.d))
        z=np.log(np.abs(self.x)/beta+1.0)*np.sign(self.x)-g/alpha
        x_sgn=np.sign(z)
        x_val=beta*np.exp(np.abs(z))-beta
        #y= np.sinh(np.arcsinh(self.x*self.d)-g/np.sqrt(self.lam))
        y=x_sgn*x_val        
        y_norm=lina.norm(y.flatten(),ord=1)
        if(y_norm>self.D):
            x_val=self.project(x_val,beta,self.D)
            self.x[:]=x_sgn*x_val        
        else:
            self.x[:]=y
        
    def project(self, y, beta, D):
        mu_idx=np.argsort(y)
        p1=np.take_along_axis(y, mu_idx, axis=0)*(D+(self.d-np.arange(self.d)+1)*beta)+beta*D
        p2=np.cumsum(np.take_along_axis(y, mu_idx, axis=0)[::-1])[::-1]
        rho = np.min(np.nonzero(p1-beta*p2>0))
        I=self.d-rho+1
        z=(p2[rho]+I*beta)/(D+I*beta) 
        return np.maximum((y+beta)/z-beta,0.0)
