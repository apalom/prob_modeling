# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:37:20 2019

@author: Alex Palomino
"""

#%% Q1





#%% Q2





#%% Q3a - Draw 30 Samples from a Gaussian(0,2)

import numpy as np
import scipy.stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Normal distribution paramterization
mu_init = 0; sd_init = 2; 

# Draw 30 samples from a normal dist N(0,2)
n=30; x = np.linspace(-5,5,n);
y0 = np.random.normal(mu_init,sd_init,n)

# Estimate mu and sigma from samples
mu_est = np.mean(y0)
sd_est = np.std(y0)

# Define likelihood function
def model(params):
    n = 30; 
    mu_in = np.array(n*[params[0]]); 
    sd_in = np.array(n*[params[1]]);
    y0 = np.random.normal(mu_in,sd_in,30)
    
    # Define Gaussian likelihood function
    LL = n/2 * np.log(2*np.pi) + (1/2) * sum(np.log(sd_in**2)) + (1/2) * sum((y0 - mu_in)**2/(sd_in**2))
    
    print('LL = {:0.2f}'.format(LL))    
    
    return LL

# Result of likelihood model minimization (i.e. maximization of negative log-likelihood)
lik_model = minimize(model, np.array([mu_est,sd_est]), method='L-BFGS-B', options = {'disp':True})
print(lik_model)

# Gaussian parameters from MLE
mu_MLE = lik_model['x'][0]; 
sd_MLE = lik_model['x'][1];

# Plot MLE results
plt.figure(figsize=(10,6))
yMLE = scipy.stats.norm.pdf(x,mu_MLE,sd_MLE)
plt.plot(x, yMLE, label='MLE')

# Scatter plot from known Gaussian distribution
yKnown = scipy.stats.norm.pdf(x,mu_init,sd_init)
plt.scatter(x, yKnown, s=5, c='black', label='Samples')

plt.title('Gaussian Samples (qty 30) $\mu=0$, $\sigma=2$')
plt.xlabel('x'); plt.ylabel('density'); 
plt.legend()
plt.show()


#%% Q3b - Draw 30 Samples from a Student-t(0,2)

import numpy as np
import scipy.stats
from scipy.special import gamma
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Student-t distribution paramterization
mu_init = 0; sd_init = 2; 

# Draw 30 samples from a normal dist N(0,2)
n=30; x = np.linspace(-10,10,n);
y0 = np.random.normal(mu_init,sd_init,n)

# Add noise to samples
#y0 =  np.append(y0, [8, 9, 10])

# Estimate mu and sigma from samples
mu_est = np.mean(y0)
sd_est = np.std(y0)
v_est = n-1;

# Define likelihood function
def model(params):
    n = 30; 
    mu = np.array(n*[params[0]]); 
    sd = np.array(n*[params[1]]); 
    v = np.array(n*[params[2]]); 
    y0 = np.random.normal(mu,sd,n)
    
    # Student-t likelihood function
    #LL = n*np.log((gamma((v+1)/2))./(np.sqrt(np.pi*(v-2))*gamma(v/2))) + (1/2) * sum(np.log(sd**2)) + ((v+1)/2) * sum(np.log(1+((y0 - mu)**2)./sd**2))
    LL = (n*np.log((gamma((v+1)/2))/(np.sqrt(np.pi*(v-2))*gamma(v/2))) +
    (1/2) * sum(np.log(sd**2)) +
    ((v+1)/2) * sum(np.log(1+((y0 - mu)**2)/sd**2)))
    
    print('LL =', LL[0])    
    
    return LL

lik_model = minimize(model, np.array([mu_est,sd_est,v_est]), method='L-BFGS-B', options = {'disp':True})
print(lik_model)

# Normal parameters from MLE
mu_MLE = lik_model['x'][0]; 
sd_MLE = lik_model['x'][1];

# Plot results
plt.figure(figsize=(10,4))
yMLE = scipy.stats.norm.pdf(x,mu_MLE,sd_MLE)
plt.plot(x, yMLE, label='MLE')

yKnown = scipy.stats.norm.pdf(x,mu_init,sd_init)
plt.scatter(x, yKnown, s=5, c='black', label='Samples')

plt.title('Gaussian Samples + Noise (qty 30) $\mu=0$, $\sigma=2$')
plt.xlabel('x'); plt.ylabel('density'); 
plt.legend()
plt.show()



