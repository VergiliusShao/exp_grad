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

Running Command  
 nohup python experiment_cem.py -d 0 --mode PP > pp.out &  
 nohup python experiment_cem.py -d 1 --mode PN > pn.out &  
 nohup python experiment_cem.py -d 0 --mode PP --smooth 0.01> pp_zo.out &  
 nohup python experiment_cem.py -d 1 --mode PN --smooth 0.01> pn_zo.out &  
 nohup python experiment_cem.py -d 0 --mode PP -s True > pp_mtx.out &  
 nohup python experiment_cem.py -d 1 --mode PN -s True> pn_mtx.out &  
 nohup python experiment_cem.py -d 0 --mode PP -s True --smooth 0.01 > pp_mtx_zo.out &  
 nohup python experiment_cem.py -d 1 --mode PN -s True --smooth 0.01> pn_mtx_zo.out &  
