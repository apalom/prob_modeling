# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:00:17 2019

@author: Alex
"""

#%% Q1 Simulation Data Set

import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import pandas as pd

w_true = [-0.3, 0.5]

n = 20;
x_n = np.random.uniform(-1,1,n)
g_Noise = np.random.normal(0,0.2,n)
y_n = n*[w_true[0]] + x_n*w_true[1] + g_Noise

plt.scatter(x_n,y_n)

#%
alpha = 2; beta = 25;
x = np.random.uniform(-1,1,size=(n,n))

w_mean = np.zeros(n).ravel()
w_cov = np.zeros((n, n))
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

