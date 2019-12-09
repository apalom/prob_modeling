# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 09:08:33 2019

@author: Alex Palomino
"""

import numpy as np
import scipy.stats as sp
from scipy.linalg import logm, expm
from scipy.stats import truncnorm
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from timeit import default_timer as timer

bank_test = pd.read_csv('bank-note/test.csv', names=['var','skew','curt','entropy','genuine']);
bank_trn = pd.read_csv('bank-note/train.csv', names=['var','skew','curt','entropy','genuine']);

phi = bank_trn[['var','skew','curt','entropy']];
phiT = phi.T
phiTphi = np.matmul(np.transpose(phi.values),phi.values);
t = bank_trn[['genuine']];
N1 = np.sum(t.values);
N0 = len(bank_trn) - N1;

bank1 = bank_trn[bank_trn['genuine']==1]
bank0 = bank_trn[bank_trn['genuine']==0]
phi1 = bank1[['var','skew','curt','entropy']];
phi0 = bank0[['var','skew','curt','entropy']];

def prior(mu, sigma):
    return np.random.multivariate_normal(mu,sigma)

def cond(z, sigma):
    cond_mu = sigma[0][1]*(sigma[1][1]**-1) * z
    cond_std = sigma[0][0] - sigma[0][1]*(sigma[1][1]**-1)*sigma[1][0]
    return np.random.normal(cond_mu, cond_std)

#%% Parameters

smpls=1000; burnin=10000; total=smpls+burnin;

# model initialization and function definition
w_init = np.zeros((1,4), dtype=float);
w = pd.DataFrame(np.zeros((total,4), dtype=float));
sig_init = np.linalg.inv(np.eye(4));
z = pd.DataFrame(np.zeros((len(bank_trn),1))); 

V = prior(np.diag(phiTphi), sig_init)*np.eye(4);
Vinv = np.linalg.inv(V)

#%% Gibbs Sampler

for i in np.arange(0,total-1):
    mu_i = pd.DataFrame(np.matmul(phi.values, w.iloc[i].T.values))
    
    z.iloc[bank1.index] = truncnorm.pdf(mu_i.iloc[bank1.index], 0, 1)
    z.iloc[bank0.index] = truncnorm.pdf(mu_i.iloc[bank0.index], 0, 1)
    
    M = np.dot(Vinv, np.array(np.dot(sig_init, w.iloc[i]) + np.matmul(phiT.values,z.values)[:,0]))
    
    w.iloc[i+1] = np.random.multivariate_normal(M,V)
    print(i)
