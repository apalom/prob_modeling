# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:42:53 2019

@author: Alex Palomino

Tensorflow is a learning framework. TF is one of the many Machine Learning libraries.
Developed and maintained by Amazon. TF and PYTORCH are the biggest/best developed.
TF is a big software packaged as a Python library. 
New data scientists must know TF (no exception), but PyTorch is growing in popularity
in academia. PyTorch is ver simple to learn (like NumPy). Could self teach in 1 hr. 
This lecture will focus on TF (developed by Google).

Core functionality... TF provides a very powerful parallel communication (CPU/GPU/TPU)
and automatic differentiation (through back-propogation). GPU is much faster.

Static (TF 1.X version) vs. Eager Mode (PyTorch, TF 2.0)
In our class we focus on TF 1.X where we must predefine tensors and computation graphs.
Lots of subtelty in working on TF 1.X. We use TF 1.X in class because it is more
robust and efficient and broadly deployed.  

A tensor: a multi-dimensional array.
A flow: a computation graph. 
So working process is a combination of tensors (values) passed through functions (flows).

Workflow... 
Define inputs and variable tensors (weights/params). Keras does this for you. 
Define computation graphs from inputs to outpus. 
Define loss function and optimizer.
Execute the graphs. 

"""

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt

print( np.__version__)
print( tf.__version__)