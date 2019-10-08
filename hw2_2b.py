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
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#% Q2 Binary Logistic & Probit Regression for Bank Note Auth

bank_test = pd.read_csv('data/hw2_bank-note/test.csv', names=['var','skew','curt','entropy','genuine'])
bank_trn = pd.read_csv('data/hw2_bank-note/train.csv', names=['var','skew','curt','entropy','genuine']);

#%% Probit Regression Update: L-BFGS Optimization

phi = np.transpose(bank_trn[['var','skew','curt','entropy']].values)
phiT = np.transpose(phi)
twoI = 2*np.identity(phi.shape[0]);
t = bank_trn[['genuine']].values

i=0; err = 100;
tol = 1E-5; maxiter = 100;

#w_old = np.array([[0.],[0.],[0.],[0.]])
w_old = np.random.normal(0,1,4).reshape(4,1)

def probitUp(w_old, *args):
    
    phi=args[0];
    t=args[1];
    
    a = np.matmul(np.transpose(w_old),phi).reshape(len(phiT),1)
    ta = np.multiply(t,a)
    
    n_pdf = sp.stats.norm.pdf(a);
    n_cdf = sp.stats.norm.cdf(ta);   
    
    g = np.divide(n_pdf,n_cdf)
    tg = np.multiply(t,g)
    gradE = np.matmul(phi,tg)
    
    h1 = np.matmul(n_pdf**2,np.transpose(n_cdf**(-2)))
    h2 = np.matmul(np.matmul(t,np.transpose(a)),n_pdf*np.transpose(n_cdf**(-1)))
    h12= h1+h2
    
    h = np.matmul(np.matmul(-phi,h12),np.transpose(phi)) + twoI
    hInv = h**(-1)
    
    update = -np.matmul(hInv,gradE)
    
    w_new = w_old - update
    
    err = np.linalg.norm(w_old - w_new)
    #w_old = w_new
    return err
    
results = sp.optimize.fmin_l_bfgs_b(probitUp, x0=w_old, args=(phi,t), approx_grad=True, maxiter=100)

w_new = results[0]

testPhi = bank_test[['var','skew','curt','entropy']]
testLabel = bank_test[['genuine']]

testLabel['pred'] = np.zeros((len(testPhi),1))
testLabel['err'] = np.zeros((len(testPhi),1))

for idx, row in testPhi.iterrows():
    if np.dot(row,w_new) <= 0:
        label = 0;
    elif np.dot(row,w_new) > 0:
        label = 1;
    
    testLabel['pred'].at[idx] = label
    
    if label == testLabel['genuine'].at[idx]:
        testLabel['err'].at[idx] = 1;
        
predErr = (len(bank_test)-testLabel['err'].sum())/len(bank_test)
print('\nSteps: ', i)
print('Pred Error: ', predErr, '\n w_new: \n', w_new)
print('\n')


#%% iterative re-weighted least squares

phi = np.transpose(bank_trn[['var','skew','curt','entropy']].values)
phiT = np.transpose(phi)
twoI = 2*np.identity(phi.shape[0]);
t = bank_trn[['genuine']].values

i=0; err = 100;
tol = 1E-5; maxiter = 100;

#w_old = np.array([[0.],[0.],[0.],[0.]])
w_old = np.random.normal(0,1,4).reshape(4,1)

while err > tol:
    
    a = np.matmul(np.transpose(w_old),phi).reshape(len(phiT),1)
    ta = np.multiply(t,a)
    
    n_pdf = sp.stats.norm.pdf(a);
    n_cdf = sp.stats.norm.cdf(ta);   
    
    g = np.divide(n_pdf,n_cdf)
    tg = np.multiply(t,g)
    gradE = np.matmul(phi,tg)
    
    h1 = np.matmul(n_pdf**2,np.transpose(n_cdf**(-2)))
    h2 = np.matmul(np.matmul(t,np.transpose(a)),n_pdf*np.transpose(n_cdf**(-1)))
    h12= h1+h2
    
    h = np.matmul(np.matmul(-phi,h12),np.transpose(phi)) + twoI
    hInv = h**(-1)
    
    update = -np.matmul(hInv,gradE)
    
    w_new = w_old - update
    
    err = np.linalg.norm(w_old - w_new)
    w_old = w_new
    
    i += 1;
    
    print('Step', i, err)
    
    if i > maxiter:
        err = 0;
        print('\n *** No-Convergence *** ')

   
testPhi = bank_test[['var','skew','curt','entropy']]
testLabel = bank_test[['genuine']]

testLabel['pred'] = np.zeros((len(testPhi),1))
testLabel['err'] = np.zeros((len(testPhi),1))

for idx, row in testPhi.iterrows():
    if np.dot(row,w_new) <= 0:
        label = 0;
    elif np.dot(row,w_new) > 0:
        label = 1;
    
    testLabel['pred'].at[idx] = label
    
    if label == testLabel['genuine'].at[idx]:
        testLabel['err'].at[idx] = 1;
        
predErr = (len(bank_test)-testLabel['err'].sum())/len(bank_test)
print('\nSteps: ', i)
print('Pred Error: ', predErr, '\n w_new: \n', w_new)
print('\n')




