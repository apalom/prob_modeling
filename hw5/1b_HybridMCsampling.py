# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:57:12 2019

@author: Alex Palomino

Implement Hybrid Monte-Carlo sampling with Leapfrog. Let us fi
x L = 10, and
vary eps from [0.005; 0.01; 0.1, 0.2; 0.5]. Run your chain. Similar to the above, 
for each setting of eps, record the acceptance rate,and draw a fi
gure showing eps
v.s. acceptance rate. What do you observe? For each setting of eps, draw the 
normalized histogram (50 bins)of collected 5K samples. What do you observe?

z(t) = position vector of ball at time t; 
r(t) = momentum vector of ball at time t;
U(z(t)) = potentional energy of ball at time t; 
K(r(t)) = kinetic energy of ball at time t;
H(z,r) = U(z) + K(r) total energy
"""

import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

def pDist(z):
    return np.exp(-z**2)*((1+np.exp(-10*z-3))**-1)
        
smpls=50000; burnin=100000; total=smpls+burnin;

tau = np.array([0.005,0.01,0.1,0.2,0.5, 1]);

t = 3;
tau1 = tau[t];

z = np.zeros(total);
accept = np.zeros(total)

s_accept = np.zeros(total);
s_accept[0] = 1;
s_reject = np.zeros(total);

unif_RV = np.random.uniform(size=total)

for i in np.arange(1,total-1):
    
    z_prime = np.random.normal(z[i],tau1);
    smpl_candidate = np.log(pDist(z_prime));
    smpl = np.log(pDist(z[i]))
    #smpl = np.log(s_accept[j])
    
    ratio = smpl_candidate - smpl;
    probability = min(0., ratio);
        
    if np.exp(probability) >= unif_RV[i]:
        z[i+1] = z_prime
        s_accept[i] = smpl_candidate;        
        accept[i] = 1;

    else:
        z[i+1] = z[i]
        s_reject[i] = smpl_candidate;

zStationary = z[burnin:];
zPicks = zStationary[0::10]

accRatio[t] = np.sum(accept[burnin:])/smpls;
print('Acceptance Ratio = %.2f' % accRatio[t])

font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)