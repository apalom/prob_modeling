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

#% plot samples
plt.style.use('ggplot')
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)
plt.figure(figsize=(8,8))

plt.scatter(smpl0[:,0], smpl0[:,1], s=8, alpha=0.3, label='[3.0,0.0]')
plt.scatter(smpl3[:,0], smpl3[:,1], s=8, alpha=0.3, label='[3.0,3.0]')
plt.scatter(smpl[:,0], smpl[:,1], s=20, edgecolor='w', color='k', label='[3.0,2.9]')

plt.title('Samples from 2d Gaussian')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()