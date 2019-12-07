# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 11:29:23 2019

@author: Alex Palomino
"""

import numpy as np
from scipy.linalg import logm, expm
import pandas as pd
import matplotlib.pyplot as plt

mu = np.array([0,0])
sigma = np.array([[3.0,2.9],[2.9,3.0]])

def pDist(mu, sig):
    return np.random.multivariate_normal(mu,sigma)

def dUdz(mu, sig,e):
    return (pDist(mu+e, sig+e)-pDist(mu+e, sig+e))/e

smpl = np.zeros((n_iters,2))

for i in range(n_iters):
  smpl[i] = np.random.multivariate_normal(mu,sigma)

df_smpl = pd.DataFrame(smpl, columns=['z1','z2'])
rho = df_smpl['z1'].corr(df_smpl['z2'])

#%% HMC Sampler

eps = 0.1; L = 20; # leapfrog steps 
M = np.eye(2); s= np.diag(M); # mass definition matrix
total = 500 # samples

z = np.empty((2, total + 1));
z[:,0] = -4; r = 0
accept = np.empty((2, total + 1));
ratio = np.empty((2, total + 1));

unif_RV = np.random.uniform(size=total)

for i in np.arange(0,total):
    
    r = np.random.normal([0,0],s) 
    Kr = (1/2)*r**2*(s**(-1))*np.eye(2) # kinetic energy
    Uz = -logm(pDist(mu, sigma)*np.eye(2)).real # potential energy
    H = Uz + Kr # total energy
    
    for steps in range(L):
                
        r_half = r - (eps/2)*dUdz(mu, sigma, eps);
        z_whole = z[:,i] + eps*(r_half/s);
        r_whole = r_half - (eps/2)*dUdz(mu, sigma, eps);
        
    z_prime = z_whole;
    r_prime = -r_whole;
        
    Kr_prime = (1/2)*r_prime**2*(s**(-1))*np.eye(2) # kinetic energy
    Uz_prime = -logm(pDist(mu, sigma)*np.eye(2)).real # potential energy
    H_prime = Uz_prime + Kr_prime # total energy
    
    ratio[:,i] = np.diag(expm(-H_prime + H));
    prob_z1 = min(1., ratio[:,i][0]);
    prob_z2 = min(1., ratio[:,i][1]);
    
    if prob_z1 >= unif_RV[i]:
        z[0,i+1] = z_prime[0]        
        accept[0,i] = 1;

    else:
        z[0,i+1] = z[0,i]
        
    if prob_z2 >= unif_RV[i]:
        z[1,i+1] = z_prime[1]        
        accept[1,i] = 1;

    else:
        z[1,i+1] = z[1,i]

#% plot samples
plt.figure(figsize=(10,6))

plt.plot(z[0,:], z[1,:], lw=1.0,c='turquoise', alpha=0.4)
plt.scatter(z[0,:], z[1,:], marker='o', c = 'k')

plt.title("Hamiltonian Sampling")
plt.xlabel("z1")
plt.ylabel("z2")
plt.show()

#% plot trace
plt.figure(figsize=(10,6))

plt.plot(z[0,:], label='z1')
plt.plot(z[1,:], label='z2')

plt.title("Parameter Trace")
plt.xlabel("Sample")
plt.ylabel("Value")
plt.legend()
plt.show()