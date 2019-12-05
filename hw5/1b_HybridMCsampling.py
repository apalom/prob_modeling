# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:57:12 2019

@author: Alex Palomino

Implement Hybrid Monte-Carlo sampling with Leapfrog. Let us fi
x L = 10, and
vary eps from [0:005; 0:01; 0:02; 0:05; 1]. Run your chain. Similar to the above, 
for each setting of eps, record the acceptance rate,and draw a fi
gure showing eps
v.s. acceptance rate. What do you observe? For each setting of eps, draw the 
normalized histogram (50 bins)of collected 5K samples. What do you observe?

"""

import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

def pDist(z):
    #sigmoid = (1/(1+np.exp(-10*z - 3)));
    #np.exp(-z**2)*sigmoid
    return np.exp(-z**2)*((1+np.exp(-10*z-3))**-1)
        
smpls=50000; burnin=100000; total=smpls+burnin;

tau = np.array([0.005, 0.01, 0.02, 0.05, 1]);

t = 3;
tau1 = tau[t];

z = np.zeros(total);
accept = np.zeros(total)

s_accept = np.zeros(total);
s_accept[0] = 1;
s_reject = np.zeros(total);

unif_RV = np.random.uniform(size=total)