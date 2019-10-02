# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:00:17 2019

@author: Alex
"""

#%% Q1 Simulation Data Set

import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

n = 20;
w_true = [-0.3, 0.5]
w_v = [np.linspace(-1,1,n), np.linspace(-1,1,n)]

x_n = np.random.uniform(-1,1,n)
g_Noise = np.random.normal(0,0.2,n)
y_n = n*[w_true[0]] + x_n*w_true[1] + g_Noise

plt.scatter(x_n,y_n)

k = np.shape(w_true)[0]; alpha = 2; beta = 25;
x = np.random.uniform(-1,1,size=(n,n))

w_mean = np.zeros(k).ravel()
w_cov = np.zeros((k, k))
np.fill_diagonal(w_cov, alpha)
w_covDet = np.linalg.det(w_cov)
w_covInv = np.linalg.inv(w_cov)

# Calculate and Plot Weight Prior 
i=0;j=0;
w_prior = np.zeros((n,n))
for w0 in w_v[0]:   
    j=0;
    for w1 in w_v[1]:
        w01 = np.array([w0,w1])
        w01t = np.transpose(w01)
        xTsigx = np.matmul(np.matmul(w01t,w_covInv),w01)      
        
        w_prior[i,j] = ((2*np.pi)**(-k/2))*(w_covDet**(-1/2))*np.exp(xTsigx)
        print(i,j, xTsigx)
        j+=1
    i+=1

ax = sns.heatmap(w_prior)
ax.set_xlabel('w1')
ax.set_ylabel('w0')

#%%

x=np.linspace(-1,1,20)

for i in range(n):
    w_0samp = np.random.choice(w_prior.flatten(), 1)[0]
    w_1samp = np.random.choice(w_prior.flatten(), 1)[0]
    y_samp = w_0samp + x*w_1samp
    plt.plot(x,y_samp, label='[{0:.3f},{1:.3f}]'.format(w_0samp,w_1samp))

plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best', prop={'size': 8})
plt.tight_layout()

#%%

w_mean = np.zeros(n).ravel()
w_cov = np.zeros((2, 2))
np.fill_diagonal(w_cov, alpha)
w_prior = sp.multivariate_normal.pdf(x, mean=w_mean, cov=w_cov)
 
lik_mean = y_n
lik_cov = np.zeros((n, n))
np.fill_diagonal(lik_cov, beta**(-1))
lik_fun = sp.multivariate_normal.pdf(x, mean=lik_mean, cov=lik_cov)

#%% Q2 Binary Logistic & Probit Regression for Bank Note Auth

bank_test = pd.read_csv('data/hw2_bank-note/test.csv', names=['var','skew','curt','entropy','genuine'])
bank_trn = pd.read_csv('data/hw2_bank-note/test.csv', names=['var','skew','curt','entropy','genuine']);


#%% Q3 Multi-Class Logistic Regression for Car Evaluation

car_test = pd.read_csv('data/hw2_car/test.csv', names=['buying','maint','doors','persons','lug_boot','safety','label']);
car_trn = pd.read_csv('data/hw2_car/test.csv', names=['buying','maint','doors','persons','lug_boot','safety','label']);

