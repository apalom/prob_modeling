# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:55:34 2019

@author: Alex Palomino

Implement Metropolis-Hasting, with Gaussian proposal, q(zn+1 | zn) = N(zn+1 | zn, tau).
Vary tau from [0:005; 0:01; 0:02; 0:05; 1]. Run your chain. For each setting of tau , 
record the acceptance rate (i.e., how many candidate samples are accepted/the total
number of candidate samples generated). Draw a fi
gure, where the x-axis represents 
the setting of tau , and y-axis the acceptance rate. What do you observe? For each
setting of tau , draw a fi
gure, show a normalized histogram of the 5K samples collected.

"""

import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

def pDist(z):
    return np.exp(-z**2)*((1+np.exp(-10*z-3))**-1)
        
smpls=50000; burnin=100000; total=smpls+burnin;

tau = np.array([0.01, 0.1, 0.2, 0.5, 1]);

t = 4;
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

#% plot chain histogram

plt.style.use('ggplot')
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)
plt.figure(figsize=(10,4))
plt.hist(zPicks, density=True, bins=50, alpha=0.75, edgecolor='white', label='MCMC')
plt.plot(x0, pz(x0), c='olive', lw=1.5, label='GQ')
#plt.fill(x0, pz(x0), x0, np.zeros(len(x0)), c='olive', alpha=0.3, label='GQ')
#plt.axvline(z0, color='k', linestyle=':', lw=1.5)

plt.title("Sampling Histogram (tau = %.3f)" % tau[t])
plt.xlim(-5,5)
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

#%% plot acceptance ratio

plt.style.use('ggplot')
plt.figure(figsize=(4,4))
plt.plot(tau, accRatio, lw=1.5, marker='o', markerfacecolor='k')

plt.title("Acceptance Rate")
plt.xlabel("tau")
plt.ylabel("Value")
plt.show()
