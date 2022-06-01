# library
A library for exponeitated update based online/stochastic optimisation algorithms and experiments on them.

## Dependencies
In order to run the experiments the following software is necessary:

| Software     | Importance           | Installation Instruction                                              |
|--------------|----------------------|-----------------------------------------------------------------------|
| Python       | Necessary            | https://wiki.python.org/moin/BeginnersGuide/Download                  |
| MXNet        | Necessary            | https://mxnet.apache.org/versions/1.7.0/get_started                  |
| Numpy        | Necessary            | https://numpy.org/install/                                            |
| SciPy        | Necessary            | https://scipy.org/ |

Running Command: python experiment_cem.py  
 -n number of samples per class  
 -t number of iterations  
 -d devices: -1 cpu, i-th gpu for i>=0  
 -a choice of algorithm: ao_exp_grad ao_exp_ftrl ao_grad ao_ftrl fista  
 -b use mini-batch  
 -g access the first order information  
 --l1 l1 regularisation   
 --l2 l2 regularisation  
 --mode: PP or PN  
 --kappa: minimum of the loss  
 --seed: random seed  
 --path: folder storing the experimental results  

