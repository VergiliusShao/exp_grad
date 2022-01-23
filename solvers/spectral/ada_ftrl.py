from scipy.optimize import OptimizeResult
import numpy as np
import numpy.linalg as lina
class AdaFTRL(object):
    def __init__(self, func, func_p, x0, D, eta=1.0):
        self.func = func
        self.func_p=func_p
        self.x= np.zeros(shape=x0.shape)
        self.x[:]= x0
        self.d = np.minimum(x0.shape[0],x0.shape[1])
        self.D=D
        self.lam=np.zeros(self.d)+1e-6
        self.theta=np.zeros(shape=x0.shape)
        self.eta=eta
        
    def update(self):
        g=self.func_p(self.x)
        self.step(g)
        return self.x

    def step(self,g):
        self.update_parameters(g)
        self.md(g)

    def update_parameters(self,g):
        s=np.linalg.svd(g,full_matrices=False,compute_uv=False)
        self.lam+=(s**2)
        self.theta-=g

    def md(self,g):
        h=np.sqrt(self.lam)*self.eta
        u,_z,v =np.linalg.svd(self.theta,full_matrices=False,compute_uv=True)
        _y=_z/h
        y_norm=lina.norm(_y.flatten(),ord=1)
        if(y_norm>self.D):
            a=1.0/np.sqrt(h)
            __y=_y/a
            _y=self.project(__y,a,self.D)*a
        self.x=np.dot(u*_y, v)
        
    def project(self, v, a, c):
        mu_idx=np.argsort(v/a)[::-1]
        p1=np.cumsum(np.take_along_axis(a*v, mu_idx, axis=0))
        p2=np.cumsum(np.take_along_axis(a**2, mu_idx, axis=0))
        p3=np.take_along_axis(v/a, mu_idx, axis=0)
        rho = np.max(np.nonzero(p1-p2*p3<c))
        theta=(p1[rho]-c)/p2[rho]	
        return np.maximum(v-theta*a,0.0)