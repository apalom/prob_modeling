# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:58:06 2019

@author: Alex Palomino
"""

import numpy as np
import scipy.stats as sp
from scipy.linalg import logm, expm
import pandas as pd
import matplotlib.pyplot as plt

bank_test = pd.read_csv('bank-note/test.csv', names=['var','skew','curt','entropy','genuine']);
bank_trn = pd.read_csv('bank-note/train.csv', names=['var','skew','curt','entropy','genuine']);

phi = bank_trn[['var','skew','curt','entropy']];
phiTphi = np.matmul(np.transpose(phi.values),phi.values);
t = bank_trn[['genuine']];

#%% HMC Parameters
eps = np.array([0.005,0.1,0.2,0.5]);
e = 0; eps1 = eps[e];

L = np.array([10, 20, 50]);
el = 0; L1 = L[el];

accRatio = pd.DataFrame(np.zeros((len(L), len(eps))), index=L, columns=eps)

# model initialization and function definition
w_init = np.zeros((1,4), dtype=float);

mu = [0., 0., 0., 0.];
sigma = np.eye(4);
M = np.eye(4); s = np.diag(M); # mass definition matrix

def prior(mu, sigma):
    return np.random.multivariate_normal(mu,sigma)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def loglikelihood(phi, t):
    y = np.zeros(len(bank_trn));
    t = np.zeros(len(bank_trn));

    for n in range(len(bank_trn)):
        w = prior(mu, sigma)
        wT = np.transpose(w)
        y[n] = sigmoid(np.matmul(wT,phi.iloc[n].values))
        t1 = t[n]
    
    return (-1/2)*np.matmul(wT, w) + np.sum(t1*np.log(y[n])+(1-t1)*np.log(1-y[n]))
     
def dUdz(mu, sig,e):
    return (prior(mu+e, sig+e)-prior(mu+e, sig+e))/e

#smpls=100000; burnin=10000; total=smpls+burnin;
smpls=1000; burnin=500; total=smpls+burnin;

z = np.empty((4, total + 1)); z[:,0] = -0;
r = np.empty((4, total + 1));
accept = np.zeros((4, total + 1));
ratio = np.empty((4, total + 1));

#%% run HMC

unif_RV = np.random.uniform(size=total)

for i in np.arange(0,total):
    
    r = np.random.normal(mu,s) 
    Kr = (1/2)*r**2*(s**(-1))*np.eye(4) # kinetic energy
    Uz = -logm((loglikelihood(phi, t))*np.eye(4)).real # potential energy
    H = Uz + Kr # total energy
    
    for steps in range(L1):
                
        r_half = r - (eps1/2)*dUdz(mu, sigma, eps1);
        z_whole = z[:,i] + eps1*(r_half/s);
        r_whole = r_half - (eps1/2)*dUdz(mu, sigma, eps1);
        
    z_prime = z_whole;
    r_prime = -r_whole.reshape(4,1);
        
    Kr_prime = (1/2)*(r_prime**2 * s**(-1)) # kinetic energy
    Uz_prime = -logm(prior(mu, sigma)*np.eye(4)).real # potential energy
    H_prime = Uz_prime + Kr_prime # total energy
    
    ratio[:,i] = np.diag(expm(-H_prime + H));
    prob_z1 = min(1., ratio[:,i][0]);
    prob_z2 = min(1., ratio[:,i][1]);
    prob_z3 = min(1., ratio[:,i][2]);
    prob_z4 = min(1., ratio[:,i][3]);
    
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
    
    if prob_z3 >= unif_RV[i]:
        z[2,i+1] = z_prime[2]        
        accept[2,i] = 1;
    else:
        z[2,i+1] = z[1,i]
        
    if prob_z4 >= unif_RV[i]:
        z[3,i+1] = z_prime[3]        
        accept[3,i] = 1;
    else:
        z[3,i+1] = z[1,i]

    print(i, z[:,i])
    
zStationary = z[burnin:];
zPicks = zStationary[0::10]

accRatio[e] = np.sum(accept[burnin:])/smpls;
print('Stepsize: ', eps[e])
print('Acceptance Ratio = %.4f' % accRatio.values[el][e])
