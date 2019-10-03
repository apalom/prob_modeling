# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:16:29 2019

@author: Alex Palomino
"""

import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#% Q2 Binary Logistic & Probit Regression for Bank Note Auth

bank_test = pd.read_csv('data/hw2_bank-note/test.csv', names=['var','skew','curt','entropy','genuine'])
bank_trn = pd.read_csv('data/hw2_bank-note/test.csv', names=['var','skew','curt','entropy','genuine']);

#%%

weights = pd.DataFrame(np.zeros((1,4)))
phi = bank_trn[['var','skew','curt','entropy']]
phiTphi = np.matmul(np.transpose(phi),phi);
t = bank_trn[['genuine']]

Hinv = np.linalg.inv(phiTphi + 2*np.identity(phi.shape[1]))

#%% 

i=0; err = 100;
tol = 1E-5

w_old = np.array([[0.],[0.],[0.],[0.]])

while err > tol:
        
    gradE = np.matmul(phiTphi,w_old) - np.matmul(np.transpose(phi),t)
    
    update = np.matmul(Hinv,gradE)
    
    w_new = w_old - update 
    
    err = np.linalg.norm(w_old - w_new)
    w_old = w_new
    
    i += 1;
    print('Step', i, err)
    #weights.iloc[i] = w_new
    