import numpy as np
import numpy.linalg as lina
from loss.logistic import logistic_loss, logistic_grad
from solvers.ada_exp_grad import AdaExpGrad
from solvers.adagrad import AdaGrad
from solvers.ada_ftrl import AdaFTRL
from solvers.ada_hu import AdaHU
from solvers.ada_exp_ftrl import AdaExpFTRL
from scipy.optimize import OptimizeResult
class Sanity_Data_Generator(object):
	def __init__(self, d = 10000):
		self.d=d
		self.w = np.zeros(d)
		self.w[np.random.permutation(np.arange(d))[:d // 100]] = np.random.uniform(-1, 1, d // 100)
    
	def nextbatch(self):
		x=np.random.uniform(-1, 1, self.d)
		p=np.random.uniform(0,1)
		threshold= 1.0 / (1.0 + np.exp(-1 * x.dot(self.w)))
		if p<=threshold:
			y=1.0
		else:
			y=-1.0
		return x,y

class Func:
    def __init__(self, data):
        self.data=data    
    def __call__(self, w):
        x,y=self.data
        return logistic_loss(w,x,y)
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
    print('iteration: ',res.nit,' regret: ', res.func,' l1: ', np.linalg.norm(res.x.flatten(),ord=1))

def reg_min(Alg,generator, maxfev=100,callback=None, epoch_size=10):
    regret=[]
    x0=np.zeros(generator.d)
    D= np.linalg.norm((generator.w).flatten(),ord=1)
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
        generator=Sanity_Data_Generator()
        regret=reg_min(ALG,generator,maxfev=maxfev,callback=callback,epoch_size=100)
        regret_avg+=(np.array(regret)/trials)
    np.savetxt('logistic_'+ALG.__name__+'.csv', regret_avg, delimiter=",")