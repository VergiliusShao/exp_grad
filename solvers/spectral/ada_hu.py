from scipy.optimize import OptimizeResult
import numpy as np
import numpy.linalg as lina
class AdaHU(object):
    FEV_PER_ITER = 1
    def __init__(self, func, func_p, x0, D, eta=1.0):
        self.func = func
        self.func_p=func_p
        self.x= np.array(x0)
        self.x[:]= x0
        self.d = np.minimum(x0.shape[0],x0.shape[1])
        self.D=D
        self.lam=0.0
        self.t=0.0
        self.theta=np.array(x0)
        
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
        s=np.linalg.svd(g,full_matrices=False,compute_uv=False)
        self.lam+=(s[0]**2)
        
        #self.lam+=g**2
       
        
    def md(self,g):
        beta=1.0/self.d
        alpha=np.sqrt(self.lam)/np.sqrt(np.log(self.d))
        u,_x,v =np.linalg.svd(self.x,full_matrices=False,compute_uv=True)
        z=np.dot(u*(alpha*np.arcsinh(_x/beta)), v)-g
        u,_z,v =np.linalg.svd(z,full_matrices=False,compute_uv=True)
        _y=beta*np.sinh(_z/alpha)
        y_norm=lina.norm(_y.flatten(),ord=1)
        if(y_norm>self.D):
            _y=_y/y_norm*self.D
        self.x=np.dot(u*_y, v)
        
