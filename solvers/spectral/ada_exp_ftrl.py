from scipy.optimize import OptimizeResult
import numpy as np
import numpy.linalg as lina
class AdaExpFTRL(object):
    def __init__(self, func, func_p, x0, D, eta=0.5):
        self.func = func
        self.func_p=func_p
        self.x=np.zeros(shape=x0.shape)
        self.x[:]= x0
        self.d = np.minimum(x0.shape[0],x0.shape[1])
        self.D=D
        self.lam=0.0
        self.theta=np.zeros(shape=x0.shape)
        self.eta=eta

    def update(self):
        g=self.func_p(self.x)
        self.step(g)
        return self.x
        

    def step(self,g):
        self.update_parameters(g)
        if self.lam>0:
            self.md(g)

    def update_parameters(self,g):
        s=np.linalg.svd(g,full_matrices=False,compute_uv=False)
        self.lam+=(s[0]**2)
        self.theta-=g
        
        
    def md(self,g):
        beta=1.0/self.d
        alpha=np.sqrt(self.lam)/np.sqrt(np.log(self.d))*self.eta
        u,_z,v =np.linalg.svd(self.theta,full_matrices=False,compute_uv=True)
        _y=beta*np.exp(_z/alpha)-beta
        y_norm=lina.norm(_y.flatten(),ord=1)
        if(y_norm>self.D):
            _y=self.project(_y,beta,self.D)
        self.x=np.dot(u*_y, v)
        
    def project(self, y, beta, D):
        mu_idx=np.argsort(y)
        p1=np.take_along_axis(y, mu_idx, axis=0)*(D+(self.d-np.arange(self.d)+1)*beta)+beta*D
        p2=np.cumsum(np.take_along_axis(y, mu_idx, axis=0)[::-1])[::-1]
        rho = np.min(np.nonzero(p1-beta*p2>0))
        I=self.d-rho+1
        z=(p2[rho]+I*beta)/(D+I*beta) 
        return np.maximum((y+beta)/z-beta,0.0)
