# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:00:17 2019

@author: Alex
"""

import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('ggplot')
plt.figure(figsize=(8,6))
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)

n = 20
w_true = np.array([[-0.3], [0.5]])
k = np.shape(w_true)[0]

mu0 = 0; std0 = 0.2;

x_n = np.vstack(np.linspace(-1,1,n));

g_Noise = np.vstack(np.random.normal(mu0,std0,n))

# Linear Model
def linearModel(x_n,w,n,noise):
    y_n = n*[w[0]] + x_n*w[1] + noise*g_Noise
    return y_n

y_n = linearModel(x_n, w_true, n, 1)
y_ntrue = linearModel(x_n, w_true, n, 0)

plt.scatter(x_n,y_n, label='Noisey Samples')
plt.plot(x_n,y_ntrue, 'k--', label='Linear Model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

#%% Calculate/Plot Likelihood

alpha = 2; beta = 25;

pxls = 5*n;
w_v = [np.linspace(-1,1,pxls), np.linspace(-1,1,pxls)]
idx0 = np.round(w_v[0],1); idx1 = np.round(w_v[1],1);

w_lik = np.zeros((pxls,pxls))

i=0;
for w1 in w_v[1]:
    j=0;
    for w0 in w_v[0]:
        w01 = np.array([w0,w1]).reshape(2,1)
        mu = linearModel(x_n,w01,n,1)
        lik_pr = sp.norm.pdf(y_n,mu,beta)
        lik_fun = 1
        for p in lik_pr:
            lik_fun = p*lik_fun 
        w_lik[i][j] = lik_fun
        j+=1;
    i+=1;

w_lik = pd.DataFrame(w_lik,index=np.round(idx0,1), columns=np.round(idx1,1))
w_lik = w_lik.sort_index(ascending=False, axis=0)
ax = sns.heatmap(w_lik)
xStar = (pxls/2)*w_true[1]; 
yStar=(pxls/2)+(pxls/2)*w_true[0]; 
plt.scatter(xStar, yStar, marker='*', s=75, color='black', label='True Weight') 
ax.set_xlabel('w0')
ax.set_ylabel('w1')
ax.set_title('Likelihood')
plt.legend(loc='best', prop={'size': 10})

#%% Calculate/Plot Prior

w0,w1 = np.meshgrid(np.linspace(-1,1,pxls),np.linspace(-1,1,pxls));
w_v = np.c_[w0.ravel(),w1.ravel()];

mu_prior = np.array([[0.0],[0.0]])
alphaI = np.array([[alpha,0],[0.0,alpha]])
alphaI_inv = np.linalg.inv(alphaI)

w_prior = sp.multivariate_normal.pdf(w_v,mu_prior.ravel(),alphaI_inv)
w_prior = w_prior.reshape(w0.shape);

w_prior = pd.DataFrame(w_prior,index=idx0, columns=idx1)
w_prior = w_prior.sort_index(ascending=False, axis=0)

ax = sns.heatmap(w_prior)
xStar = (pxls/2)*w_true[1]; 
yStar=(pxls/2)+(pxls/2)*w_true[0]; 
plt.scatter(xStar, yStar, marker='*', s=75, color='black', label='True Weight') 
ax.set_xlabel('w0')
ax.set_ylabel('w1')
ax.set_title('Prior')
plt.legend(loc='best', prop={'size': 10})

#%% Plot Weight Prior Samples

plt.style.use('ggplot')
plt.figure(figsize=(8,6))
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)
x=np.linspace(-1,1,20)

for i in range(n):
    w_0samp = np.random.choice(w_prior.values.flatten(), 1)[0]
    w_1samp = np.random.choice(w_prior.values.flatten(), 1)[0]
    y_samp = w_0samp + x*w_1samp
    plt.plot(x,y_samp, label='[{0:.3f}, {1:.3f}]'.format(w_0samp,w_1samp))

plt.plot(x_n,y_ntrue, 'k--', label='Linear Model')

plt.xlabel('x')
plt.xlim((-1,1))
plt.ylabel('y')
plt.ylim((-1,1))
plt.legend(loc='best', prop={'size': 8}, title='Weights')
plt.title('Prior Samples')
plt.tight_layout()

#%% Calculate/Plot Posterior

w_post = np.multiply(w_prior,w_lik);

ax = sns.heatmap(w_post)
xStar = (pxls/2)*w_true[1]; 
yStar=(pxls/2)+(pxls/2)*w_true[0]; 
plt.scatter(xStar, yStar, marker='*', s=75, color='black', label='True Weight') 
ax.set_xlabel('w0')
ax.set_ylabel('w1')
ax.set_title('Posterior')
plt.legend(loc='best', prop={'size': 10})

#%% Plot Weight Posterior Samples

plt.style.use('ggplot')
plt.figure(figsize=(8,6))
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)
x=np.linspace(-1,1,20)

for i in range(n):
    w_0samp = np.random.choice(w_post.values.flatten(), 1)[0]
    w_1samp = np.random.choice(w_post.values.flatten(), 1)[0]
    y_samp = w_0samp + x*w_1samp
    plt.plot(x,y_samp, label='[{0:.3f}, {1:.3f}]'.format(w_0samp,w_1samp))

plt.plot(x_n,y_ntrue, 'k--', label='Linear Model')

plt.xlabel('x')
plt.xlim((-1,1))
plt.ylabel('y')
plt.ylim((-1,1))
plt.legend(loc='best', prop={'size': 8}, title='Weights')
plt.title('Posterior Samples')
plt.tight_layout()

#%%

import pymc3 as pm

alpha = 2
mu=0; mu_prior = np.array([[0.0],[0.0]])
alpha=2; alphaI = np.array([[alpha,0],[0.0,alpha]])

s = 5;
obsX = x_n[s-1]; obsY = y_n[s-1]

with pm.Model() as model:

    m_prior = pm.Normal('prior', mu=0, sigma=alpha)
    
    y_est = pm.Normal('lik', mu=0, sigma=alpha, observed=y_n[0:s])

    y_pred = pm.Normal('y_pred', mu=mu, sigma=alpha)
        
    trace = pm.sample(25, progressbar=True)

#%%
    
posterior_counts, posterior_bins = np.histogram(trace['prior'])

plt.hist(trace['prior'])

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
w_v = np.round([np.linspace(-1,1,n), np.linspace(-1,1,n)],4)

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

w_prior = pd.DataFrame(w_prior,index=w_v[0], columns=w_v[1])
ax = sns.heatmap(w_prior)
ax.set_xlabel('w1')
ax.set_ylabel('w0')

#%% Plot Weight Prior Samples

plt.style.use('ggplot')
plt.figure(figsize=(10,8))
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)
x=np.linspace(-1,1,20)

for i in range(n):
    w_0samp = np.random.choice(w_prior.values.flatten(), 1)[0]
    w_1samp = np.random.choice(w_prior.values.flatten(), 1)[0]
    y_samp = w_0samp + x*w_1samp
    plt.plot(x,y_samp, label='[{0:.3f}, {1:.3f}]'.format(w_0samp,w_1samp))

plt.xlabel('x')
#plt.xlim((-1,1))
plt.ylabel('y')
#plt.ylim((-1,1))
plt.legend(loc='best', prop={'size': 12}, title='Weights')
plt.tight_layout()

#%% Calculate Likelihood
import random
plt.figure(figsize=(10,8))

w_cov = np.zeros((k, k))
np.fill_diagonal(w_cov, 1/beta)
w_covDet = np.linalg.det(w_cov)
w_covInv = np.linalg.inv(w_cov)

# Posterior Dist of w given (x1,y1)
i=0;j=0;
w_lik = np.zeros((n,n))
for w0 in w_v[0]:   
    j=0;
    for w1 in w_v[1]:
        #s = random.randint(0, 1); # for 1c
        #s = np.random.choice(5,1); # for 1d
        s = np.random.choice(20,1); # for 1e
        w01 = np.array([w0-y_n[s],w1-y_n[s]])
        w01t = np.transpose(w01)
        xTsigx = np.matmul(np.matmul(w01t,w_covInv),w01)      
        
        w_lik[i,j] = ((2*np.pi)**(-k/2))*(w_covDet**(-1/2))*np.exp((-1/2)*xTsigx)
        print(i,j,w01)
        j+=1
    i+=1

w_lik = pd.DataFrame(w_lik,index=w_v[0], columns=w_v[1])
ax = sns.heatmap(w_lik)
ax.set_xlabel('w1')
ax.set_ylabel('w0')

#%% Calculate Posterior = w_lik * w_prior

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, aspect='equal')

w_post = np.dot(w_lik, w_prior)
w_post = pd.DataFrame(w_post,index=w_v[0], columns=w_v[1])

sns.heatmap(w_post)
yStar=(n/2)+(n/2)*w_true[0]+0.1; xStar = (n/2)+(n/2)*w_true[1]-0.3; 
plt.scatter(xStar, yStar, marker='*', s=100, color='white') 

ax.set_xlabel('w1')
ax.set_ylabel('w0')
print(xStar, yStar)

#%% Plot Weight Posterior Samples

plt.style.use('ggplot')
plt.figure(figsize=(10,8))
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)
x=np.linspace(-1,1,20)

for i in range(n):
    w_0samp = np.random.choice(w_post.values.flatten(), 1)[0]
    w_1samp = np.random.choice(w_post.values.flatten(), 1)[0]
    y_samp = w_0samp + x*w_1samp
    plt.plot(x,y_samp, label='[{0:.3f}, {1:.3f}]'.format(w_0samp,w_1samp))

plt.xlabel('x')
plt.xlim((-1,1))
plt.ylabel('y')
plt.ylim((-1,1))
plt.legend(loc='best', prop={'size': 12}, title='Weights')
plt.tight_layout()





#%% Q2 Binary Logistic & Probit Regression for Bank Note Auth

bank_test = pd.read_csv('data/hw2_bank-note/test.csv', names=['var','skew','curt','entropy','genuine'])
bank_trn = pd.read_csv('data/hw2_bank-note/test.csv', names=['var','skew','curt','entropy','genuine']);


#%% Q3 Multi-Class Logistic Regression for Car Evaluation

car_test = pd.read_csv('data/hw2_car/test.csv', names=['buying','maint','doors','persons','lug_boot','safety','label']);
car_trn = pd.read_csv('data/hw2_car/test.csv', names=['buying','maint','doors','persons','lug_boot','safety','label']);

