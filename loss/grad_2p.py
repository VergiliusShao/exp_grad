import numpy as np
class Grad_2p:
    def __init__(self, func, n,delta):
        self.func=func
        self.n=n
        self.delta=delta
    def __call__(self, x):
        tilde_f_x_r= self.func(x)
        d = x.size
        g=np.zeros(shape=x.shape)
        for i in range(self.n):
            v= np.random.normal(size=x.shape)
            v_norm= np.linalg.norm(v)
            v=v/v_norm
            tilde_f_x_l = self.func(x+self.delta*v)
            g+=d/self.delta/self.n*(tilde_f_x_l-tilde_f_x_r)*v
        return g