# Optimisationpy
A library for optimizers currently in the context of neural networks using Python and MXNet.

## Dependencies
In order to use Optimisationpy the following software is necessary:

| Software     | Importance           | Installation Instruction                                              |
|--------------|----------------------|-----------------------------------------------------------------------|
| Python       | Necessary            | https://wiki.python.org/moin/BeginnersGuide/Download                  |
| MXNet        | Necessary            | https://mxnet.apache.org/versions/1.7.0/get_started?                  |
| Numpy        | Necessary            | https://numpy.org/install/                                            |
| Scikit-Learn | For some experiments | https://scikit-learn.org/stable/install.html#installation-instruction |

Running Command
nohup python experiment_cem.py -d 0 --mode PP > pp.out &
nohup python experiment_cem.py -d 1 --mode PN > pn.out &
nohup python experiment_cem.py -d 0 --mode PP --smooth 0.01> pp_zo.out &
nohup python experiment_cem.py -d 1 --mode PN --smooth 0.01> pn_zo.out &
nohup python experiment_cem.py -d 0 --mode PP -s True > pp_mtx.out &
nohup python experiment_cem.py -d 1 --mode PN -s True> pn_mtx.out &
nohup python experiment_cem.py -d 0 --mode PP -s True --smooth 0.01 > pp_mtx_zo.out &
nohup python experiment_cem.py -d 1 --mode PN -s True --smooth 0.01> pn_mtx_zo.out &
