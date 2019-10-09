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
plt.xlim((-1,1))
plt.ylabel('y')
plt.ylim((-1,1))
plt.legend()

#%% Calculate/Plot Prior

alpha = 2; beta = 25;

pxls = n+1;
w0,w1 = np.meshgrid(np.linspace(-1,1,pxls),np.linspace(-1,1,pxls));
w_v = np.c_[w0.ravel(),w1.ravel()];

idx0 = np.round(np.linspace(-1,1,pxls),1); 
idx1 = np.round(np.linspace(-1,1,pxls),1);

mu_prior = np.array([[0.],[0.]])
alphaI = np.array([[alpha,0],[0.0,alpha]])
alphaI_inv = np.linalg.inv(alphaI)

w_prior = sp.multivariate_normal.pdf(w_v,mu_prior.ravel(),alphaI_inv)
w_prior = w_prior.reshape(w0.shape);

w_prior = pd.DataFrame(w_prior,index=idx0, columns=idx1)
#w_prior = w_prior.sort_index(ascending=False, axis=0)

plt.style.use('ggplot')
plt.figure(figsize=(10,8))
ax = sns.heatmap(w_prior, cmap="Reds")
xStar = (pxls/2)+(pxls/2)*w_true[1]; 
yStar = (pxls/2)+(pxls/2)*w_true[0]; 
plt.scatter(xStar, yStar, marker='*', s=75, color='black', label='True Weight') 
ax.set_xlabel('w1')
ax.set_ylabel('w0')
ax.set_title('Prior')
plt.legend(loc='best', prop={'size': 10})

#%% Plot Weight Prior Samples

plt.style.use('ggplot')
plt.figure(figsize=(8,7.5))
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

#%% Calculate/Plot Poserior

alpha = 2; beta = 25;

# Add observations
pts = 20;
obs = np.concatenate((x_n[0:pts], y_n[0:pts]), axis=1)
true = np.concatenate((x_n[0:pts], y_ntrue[0:pts]), axis=1)

sNinv = alphaI + beta*np.matmul(obs.T, obs)
sN = np.linalg.inv(sNinv)

mN = beta*np.matmul(np.matmul(sN,obs.T),true[:,1])
#mN = beta*np.matmul(sN,obs.T)*true[0,1]

w_post = sp.multivariate_normal.pdf(w_v,mN.ravel(),sN)
w_post = w_post.reshape(w0.shape);

w_post = pd.DataFrame(w_post,index=idx0, columns=idx1)
w_post = w_post.sort_index(ascending=False, axis=0)

plt.style.use('ggplot')
plt.figure(figsize=(10,8))
ax = sns.heatmap(w_post, cmap="Blues")
xStar = (pxls/2)+(pxls/2)*w_true[1]; 
yStar = (pxls/2)+(pxls/2)*w_true[0];
plt.scatter(xStar, yStar, marker='*', s=75, color='black', label='True Weight') 
ax.set_xlabel('w0')
ax.set_ylabel('w1')
ax.set_title('Posterior given {} observation'.format(pts))
plt.legend(loc='best', prop={'size': 10})

#%% Plot Weight Posterior Samples

plt.style.use('ggplot')
plt.figure(figsize=(8,7.5))
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)
x=np.linspace(-1,1,20)

for i in range(n):
    w_0samp = np.random.choice(w_post.values.flatten(), 1)[0]
    w_1samp = np.random.choice(w_post.values.flatten(), 1)[0]
    y_samp = w_0samp + x*w_1samp
    plt.plot(x,y_samp, label='[{0:.3f}, {1:.3f}]'.format(w_0samp,w_1samp))

for j in np.arange(len(true)):
    #plt.scatter(obs[j,0], obs[j,1], s=50, c='k', edgecolor='w', label='[{0:.3f}, {1:.3f}]'.format(true[j,0],true[j,1]))
    plt.scatter(obs[j,0], obs[j,1], s=50, c='k', linewidth=2, edgecolor='w')

plt.plot(x_n,y_ntrue, 'k--', label='Linear Model')

plt.xlabel('x')
plt.xlim((-1.25,1.25))
plt.ylabel('y')
plt.ylim((-1.25,1.25))
plt.legend(loc='best', prop={'size': 8}, title='Weights')
plt.title('Posterior Samples')
plt.tight_layout()
