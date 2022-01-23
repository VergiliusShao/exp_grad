import numpy as np
from numpy.core.fromnumeric import argmax
import numpy.linalg as lina
from loss.logistic import logistic_loss, logistic_grad
from solvers.spectral.ada_exp_grad import AdaExpGrad
from solvers.spectral.ada_grad import AdaGrad
from solvers.spectral.ada_ftrl import AdaFTRL
from solvers.spectral.ada_hu import AdaHU
from solvers.spectral.ada_exp_ftrl import AdaExpFTRL
from scipy.optimize import OptimizeResult
from scipy.stats import ortho_group 
from loss.logistic_multi import logistic_loss, logistic_grad,sigmoid
class Sanity_Data_Generator(object):
    def __init__(self, d = 100,k=25,r=5):
        self.d=d
        self.k=k
        self.r=r
        s=np.zeros(k)
        s[np.random.permutation(np.arange(k))[:r]] = np.random.uniform(0, 10, r)
        smat=np.zeros(shape=(d,k))
        smat[:k, :k] = np.diag(s)
        u=ortho_group.rvs(d)
        v=ortho_group.rvs(k)
        self.w= np.dot(u, np.dot(smat, v))
        s=np.linalg.svd(self.w,full_matrices=False,compute_uv=False)
        self.D=np.linalg.norm(s.flatten(),ord=1)

    def nextbatch(self):
        x=np.random.uniform(-1, 1, size=(self.d,self.k))
        threshold=sigmoid(np.sum(x*self.w,axis=0))
        y=np.zeros(shape=threshold.shape)
        for i in range(y.size):
            p=np.random.uniform(0,1)
            if p<=threshold[i]:
                y[i]=1.0
            else:
                y[i]=-1.0
        return x,y
        
class Func:
    def __init__(self, data):
        self.data=data    
    def __call__(self, w):
        x,y=self.data
        return np.sum(logistic_loss(w,x,y))
    def set_data(self, data):
        self.data=data
class Func_p:
    def __init__(self, data):
        self.data=data    
    def __call__(self, w):
        x,y=self.data
        return logistic_grad(w,x,y)
    def set_data(self,data):
        self.data=data


def callback(res):
    
    s=np.linalg.svd(res.x,full_matrices=False,compute_uv=False)
    print('iteration: ',res.nit,' regret: ', res.func,' l1: ', np.linalg.norm(s,ord=1))

def reg_min(Alg,generator, maxfev=100,callback=None, epoch_size=10):
    regret=[]
    x0=np.zeros((generator.d,generator.k))
    D= generator.D
    print('target l1: ', D)
    print('target dim: ', generator.d)
    data=generator.nextbatch()
    func=Func(data)
    func_p=Func_p(data)
    alg=Alg(func=func,func_p=func_p,x0=x0,D=D)
    nit=maxfev
    fev=1
    x=x0
    reg=0.0
    regret.append(reg)
    while fev <= maxfev-1:
        reg+=(func(x)-func(generator.w))
        regret.append(reg)
        x=alg.update()
        if callback is not None and fev%epoch_size==0:
                res=OptimizeResult(func=reg, x=x, nit=fev,
                          nfev=fev, success=(x is not None))
                callback(res)
        fev+=1
        data=generator.nextbatch()
        func.set_data(data)
        func_p.set_data(data)

    return regret

trials=20
maxfev=10000
for ALG in [AdaGrad,AdaFTRL,AdaHU,AdaExpGrad,AdaExpFTRL]:
    regret_avg=np.zeros(maxfev)
    for t in range(trials):
        np.random.seed(48+t)
        generator=Sanity_Data_Generator(d=1000,k=100,r=5)
        generator.nextbatch()
        regret=reg_min(ALG,generator,maxfev=maxfev,callback=callback,epoch_size=100)
        regret_avg+=(np.array(regret)/trials)
    np.savetxt('logistic_'+ALG.__name__+'_matrix'+'.csv', regret_avg, delimiter=",")