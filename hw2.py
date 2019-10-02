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

plt.style.use('ggplot')
plt.figure(figsize=(10,8))
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)

n = 21;
w_true = [-0.3, 0.5]
w_v = [np.linspace(-1,1,n), np.linspace(-1,1,n)]

x_n = np.random.uniform(-1,1,n)
g_Noise = np.random.normal(0,0.2,n)
y_n = n*[w_true[0]] + x_n*w_true[1] + g_Noise

#plt.scatter(x_n,y_n)

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
        
        w_prior[i,j] = ((2*np.pi)**(-k/2))*(w_covDet**(-1/2))*np.exp((-1/2)*xTsigx)
        j+=1
    i+=1


ax = sns.heatmap(w_prior)
ax.set_xlabel('w1')
ax.set_xticks(np.arange(0, 22, 2))
ax.set_xticklabels(np.round(np.arange(-1, 1.2, 0.2),2))
ax.set_ylabel('w0')
ax.set_yticks(np.arange(0, 22, 2))
ax.set_yticklabels(np.round(np.arange(-1, 1.2, 0.2),2))

#%% Plot Weight Prior Samples

plt.style.use('ggplot')
plt.figure(figsize=(10,8))
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)
x=np.linspace(-1,1,20)

for i in range(n):
    w_0samp = np.random.choice(w_prior.flatten(), 1)[0]
    w_1samp = np.random.choice(w_prior.flatten(), 1)[0]
    y_samp = w_0samp + x*w_1samp
    plt.plot(x,y_samp, label='[{0:.3f}, {1:.3f}]'.format(w_0samp,w_1samp))

plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best', prop={'size': 12})
plt.tight_layout()

#%% Calculate Likelihood
plt.figure(figsize=(10,8))

w_cov = np.zeros((k, k))
np.fill_diagonal(w_cov, 1/beta)
w_covDet = np.linalg.det(w_cov)
w_covInv = np.linalg.inv(w_cov)

i=0;j=0;
w_lik = np.zeros((n,n))
for w0 in w_v[0]:   
    j=0;
    for w1 in w_v[1]:
        w01 = np.array([w0-y_n[j],w1-y_n[j]])
        w01t = np.transpose(w01)
        xTsigx = np.matmul(np.matmul(w01t,w_covInv),w01)      
        
        w_lik[i,j] = ((2*np.pi)**(-k/2))*(w_covDet**(-1/2))*np.exp((-1/2)*xTsigx)
        print(i,j,w01)
        j+=1
    i+=1

ax = sns.heatmap(w_lik)
ax.set_xlabel('w1')
ax.set_xticks(np.arange(0, 22, 2))
ax.set_xticklabels(np.round(np.arange(-1, 1.2, 0.2),2))
ax.set_ylabel('w0')
ax.set_yticks(np.arange(0, 22, 2))
ax.set_yticklabels(np.round(np.arange(-1, 1.2, 0.2),2))

#%% Calculate Posterior = w_lik * w_prior

plt.figure(figsize=(10,8))

w_post = np.dot(w_lik, w_prior)

ax = sns.heatmap(w_post)
ax.scatter(0,1, marker='*', s=100, color='yellow') 

ax.set_xlabel('w1')
ax.set_xticks(np.arange(0, 22, 2))
ax.set_xticklabels(np.round(np.arange(-1, 1.2, 0.2),2))
ax.set_ylabel('w0')
ax.set_yticks(np.arange(0, 22, 2))
ax.set_yticklabels(np.round(np.arange(-1, 1.2, 0.2),2))

#%% Plot Weight Posterior Samples

plt.style.use('ggplot')
plt.figure(figsize=(10,8))
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)
x=np.linspace(-1,1,20)

for i in range(n):
    w_0samp = np.random.choice(w_prior.flatten(), 1)[0]
    w_1samp = np.random.choice(w_prior.flatten(), 1)[0]
    y_samp = w_0samp + x*w_1samp
    plt.plot(x,y_samp, label='[{0:.3f}, {1:.3f}]'.format(w_0samp,w_1samp))

plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best', prop={'size': 12})
plt.tight_layout()


#%% Q2 Binary Logistic & Probit Regression for Bank Note Auth

bank_test = pd.read_csv('data/hw2_bank-note/test.csv', names=['var','skew','curt','entropy','genuine'])
bank_trn = pd.read_csv('data/hw2_bank-note/test.csv', names=['var','skew','curt','entropy','genuine']);


#%% Q3 Multi-Class Logistic Regression for Car Evaluation

car_test = pd.read_csv('data/hw2_car/test.csv', names=['buying','maint','doors','persons','lug_boot','safety','label']);
car_trn = pd.read_csv('data/hw2_car/test.csv', names=['buying','maint','doors','persons','lug_boot','safety','label']);

