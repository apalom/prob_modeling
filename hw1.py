# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:37:20 2019

@author: Alex Palomino
"""

#%% Q1 Student-t and Normal Distributions

import numpy as np
from scipy.stats import t
from scipy.stats import norm
import matplotlib.pyplot as plt

#plt.figure(figsize=(6,6))

x = np.linspace(-5,5,5000)

dofs = [0.1, 1,10,100,10e6]

for v in dofs:
    lab= r'$ \nu $='+str(v)
    plt.plot(x, t.pdf(x,v), lw=4, alpha=0.6, label=lab)    

plt.plot(x, norm.pdf(x,0,1), 'w--',  markersize=0.1, label=r'$N(0,1)$')

plt.xlabel('x'); plt.ylabel('density');
plt.legend()
plt.title("Student-$t$ Distributions")
plt.tight_layout()
plt.show() 

#%% Q2 Beta Distributions

import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

fig = plt.figure()
#ax = fig.add_subplot(111)    # The big subplot
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

x = np.linspace(-0.1,2,5000)
ax1.plot(x, beta.pdf(x,1,1), lw=1.5, alpha=0.6, label=r'($ \alpha $, $ \beta$)=(1,1)')
ax1.plot(x, beta.pdf(x,5,5), lw=1.5, alpha=0.6, label=r'($ \alpha $, $ \beta$)=(5,5)')
ax1.plot(x, beta.pdf(x,10,10), lw=1.5, alpha=0.6, label=r'($ \alpha $, $ \beta$)=(10,10)')
ax1.set_ylabel('density');
ax1.legend()
ax1.title.set_text('Beta Distributions')

ax2.plot(x, beta.pdf(x,1,2), lw=1.5, alpha=0.6, label=r'($ \alpha $, $ \beta$)=(1,2)')
ax2.plot(x, beta.pdf(x,5,6), lw=1.5, alpha=0.6, label=r'($ \alpha $, $ \beta$)=(5,6)')
ax2.plot(x, beta.pdf(x,10,11), lw=1.5, alpha=0.6, label=r'($ \alpha $, $ \beta$)=(10,11)')
ax2.set_xlabel('x'); ax2.set_ylabel('density');
ax2.legend()

plt.tight_layout()
plt.show() 

#%% Q3a - Draw 30 Samples from a Gaussian(0,2)

import numpy as np
import scipy.stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Normal distribution paramterization
mu_init = 0; sd_init = 2; 

# Draw 30 samples from a normal dist N(0,2)
n=30; x = np.linspace(-10,10,n);
y0 = np.random.normal(mu_init,sd_init,n)

# Add noise to samples
y0 =  np.append(y0, [8, 9, 10])

# Estimate mu and sigma from samples
mu_est = np.mean(y0)
sd_est = np.std(y0)

# Define likelihood function
def model(params):
    n = 33; 
    mu_in = np.array(n*[params[0]]); 
    sd_in = np.array(n*[params[1]]);
    y0 = np.random.normal(mu_in,sd_in,n)
#    mu_in = params[0]; 
#    sd_in = params[1];
#    y0 = np.random.normal(mu_in,sd_in,1)
    
    # Define Gaussian likelihood function
    LL = (n/2 * np.log(2*np.pi) + 
          (1/2) * sum(np.log(sd_in**2)) + 
          (1/2) * sum((y0 - mu_in)**2/(sd_in**2)))
    
    print('LL = {:0.2f}'.format(LL))    
    
    return LL

# Result of likelihood model minimization (i.e. maximization of negative log-likelihood)
lik_model = minimize(model, np.array([mu_est,sd_est]), method='L-BFGS-B', options = {'disp':True})
print(lik_model)

# Gaussian parameters from MLE
mu_MLE = lik_model['x'][0]; 
sd_MLE = lik_model['x'][1];

# Plot MLE results
plt.figure(figsize=(10,4))
yMLE = scipy.stats.norm.pdf(x,mu_MLE,sd_MLE)
plt.plot(x, yMLE, label='MLE')

# Scatter plot from known Gaussian distribution
yKnown = scipy.stats.norm.pdf(x,mu_init,sd_init)
plt.scatter(x, yKnown, s=5, c='black', label='Samples')

plt.title('Gaussian Distribution with Noisy Gaussian Samples (qty 33) $\mu=0$, $\sigma=2$')
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
n=300; x = np.linspace(-10,10,n);
y0 = np.random.normal(mu_init,sd_init,n)

# Add noise to samples
#y0 =  np.append(y0, [8, 9, 10])

# Estimate mu and sigma from samples
mu_est = np.mean(y0)
sd_est = np.std(y0)
v_est = n-1;

# Define likelihood function
def model(params):
    n = 300; 
    mu = params[0]; 
    sd = params[1];
    v = params[2];
    y0 = np.random.normal(mu,sd,1)
    
    # Student-t likelihood function
    #LL = n*np.log((gamma((v+1)/2))./(np.sqrt(np.pi*(v-2))*gamma(v/2))) + (1/2) * sum(np.log(sd**2)) + ((v+1)/2) * sum(np.log(1+((y0 - mu)**2)./sd**2))
    LL = (n*np.log((gamma((v+1)/2))/(np.sqrt(np.pi*(v-2))*gamma(v/2))) +
    (1/2) * (np.log(sd**2)) +
    ((v+1)/2) * (np.log(1+((y0 - mu)**2)/sd**2)))
    
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

plt.title('Student-t Distribution with Gaussian Samples (qty 300) $\mu=0$, $\sigma=2$')
plt.xlabel('x'); plt.ylabel('density'); 
plt.legend()
plt.show()



