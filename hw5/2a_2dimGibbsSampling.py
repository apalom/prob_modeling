# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:16:33 2019

@author: Alex Palomino

Draw 500 samples from this distribution and show the scatter plot. What do you
observe?

"""

import numpy as np
import matplotlib.pyplot as plt

mu = [0,0]
sigma = [[3.0,2.9],[2.9,3.0]]
sigma0 = [[3.0,0],[0,3.0]]
sigma3 = [[3.0,3.0],[3.0,3.0]]

sIter=500;
smpl = np.zeros((sIter,2))
smpl0 = np.zeros((sIter,2))
smpl3 = np.zeros((sIter,2))

for i in range(sIter):
  smpl[i] = np.random.multivariate_normal(mu,sigma)
  smpl0[i] = np.random.multivariate_normal(mu,sigma0)
  smpl3[i] = np.random.multivariate_normal(mu,sigma3)

df_smpl = pd.DataFrame(smpl, columns=['z1','z2'])
rho = df_smpl['z1'].corr(df_smpl['z2'])

#%% plot samples
plt.style.use('ggplot')
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)
plt.figure(figsize=(8,8))

plt.scatter(smpl0[:,0], smpl0[:,1], s=8, alpha=0.3, label='[3.0,0.0]')
plt.scatter(smpl3[:,0], smpl3[:,1], s=8, alpha=0.3, label='[3.0,3.0]')
plt.scatter(smpl[:,0], smpl[:,1], s=20, edgecolor='w', color='k', label='[3.0,2.9]')

plt.title('Samples from 2d Gaussian')
plt.xlabel('z1')
plt.ylabel('z2')
plt.legend()

#%% plot Parameter Distribution
plt.style.use('ggplot')
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)
plt.figure(figsize=(8,4))

plt.hist(smpl[:,0], bins=np.arange(-5,5,0.5), density=True, 
         edgecolor='white', label='z1')
plt.hist(smpl[:,1], bins=np.arange(-5,5,0.5), density=True, 
         alpha=0.5, edgecolor='white', label='z2')

plt.title('Parameter Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

print('z1 mean = ', np.mean(smpl[:,0]), '| z1 std = ', np.std(smpl[:,0]))
print('z2 mean = ', np.mean(smpl[:,1]), '| z2 std = ', np.std(smpl[:,1]))

#%% Implement Gibb's Sampling

import pandas as pd

df_smpl = pd.DataFrame(smpl, columns=['z1','z2'])
  
#% initialize parameter trace
n_iters = 5000;
z1, z2 = np.empty((2, n_iters + 1));
trace = np.zeros((n_iters, 2))
z1[0], z2[0] = -4, -4;
    
def cond(z, sigma):
    cond_mu = sigma[0][1]*(sigma[1][1]**-1) * z
    cond_std = sigma[0][0] - sigma[0][1]*(sigma[1][1]**-1)*sigma[1][0]
    return np.random.normal(cond_mu, cond_std)

# Sample from conditionals
for i in range(n_iters):
    
    z1[i+1] = cond(z2[i], sigma)
    z2[i+1] = cond(z1[i], sigma)
    


#%% plot samples
plt.figure(figsize=(10,6))

pts = 5000;
plt.plot(z1[:pts], z2[:pts], lw=1.5, alpha=0.4)
plt.scatter(z1[:pts], z2[:pts], marker='o', c = 'k')

plt.title("Gibbs Sampling")
plt.xlabel("z1")
plt.ylabel("z2")

plt.show()

#%% plot trace
plt.figure(figsize=(10,6))

plt.plot(z1[:pts], label='z1')
plt.plot(z2[:pts], label='z2')

plt.title("Parameter Trace")
plt.xlabel("Sample")
plt.ylabel("Value")
plt.legend()
plt.show()


#%%    
    # sample z1 given z2
    z1[i+1] = form1(z2[i])
 
    # sample z2 given z1
    z2[i+1] = ((1 - rho**2)**0.5)*np.random.randn(1, 1) + rho*z1[i];