# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:21:21 2019

@author: Alex

Implement Newton-Raphson scheme to find the MAP estimation of the feature
weights in the logistic regression model. Set the maximum number of 
iterations to $100$ and the tolerance level to be $1e-5$,  
\ie when the norm of difference between the weight vectors after one 
update is below the tolerance level, we consider it converges and stop 
updating the weights any more. Initially, you can set all the weights 
to be zero. Report the prediction accuracy on the test data. Now set 
the initial weights values be to be randomly generated, say, from the 
standard Gaussian, run and test your algorithm. What do you observe? Why?

Implement Newton-Raphson scheme to find the MAP estimation for 
Probit regression. Report the prediction accuracy 
"""

import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#% Q2 Binary Logistic & Probit Regression for Bank Note Auth

bank_test = pd.read_csv('data/hw2_bank-note/test.csv', names=['var','skew','curt','entropy','genuine'])
bank_trn = pd.read_csv('data/hw2_bank-note/train.csv', names=['var','skew','curt','entropy','genuine']);

#%% iterative re-weighted least squares

phi = bank_trn[['var','skew','curt','entropy']].values
phiT = np.transpose(phi)
phiTphi = np.matmul(phiT,phi);
t = bank_trn[['genuine']]

Hinv = np.linalg.inv(phiTphi + 2*np.identity(phi.shape[1]))

i=0; err = 100;
tol = 1E-5

w_old = np.array([[0.],[0.],[0.],[0.]])
#w_old = np.random.normal(0,1,4).reshape(4,1)

twoI = 2*np.identity(phi.shape[1]);

while err > tol:
    
    n_pdf = np.random.normal()
    n_cdf = sp.norm.cdf(n_pdf)    
    
    g1 = np.matmul(phiT,t)*(n_pdf/n_cdf)
    g2 = np.matmul(phiT,(1-t))*(n_pdf/(1-n_cdf))
    g3 = np.matmul(twoI,w_old)
    
    gradE = -(g1+g2)+g3
    
    h1 = np.matmul(phiT,t)*(twoI/(np.sqrt(2*np.pi)*n_cdf) + twoI*(n_pdf**2)/(n_cdf**2))
    h2 = np.matmul(phiT,(1-t))*(twoI/(np.sqrt(2*np.pi)*(1-n_cdf))-twoI*(n_pdf**2)/(n_cdf**2))
    h3 = twoI
    h = -(h1-h2)+h3
    
    Hinv = np.linalg.inv(h)
    
    update = np.matmul(Hinv,gradE)
    
    w_new = w_old - update 
    
    err = np.linalg.norm(w_old - w_new)
    w_old = w_new
    
    i += 1;
    
    print('Step', i, err)
    
    if i > 100:
        err = 0;
        print('\n *** No-Convergence *** ')






