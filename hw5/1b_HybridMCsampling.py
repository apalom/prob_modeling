# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:57:12 2019

@author: Alex Palomino

Implement Hybrid Monte-Carlo sampling with Leapfrog. Let us fi
x L = 10, and
vary eps from [0.005; 0.01; 0.1, 0.2; 0.5]. Run your chain. Similar to the above, 
for each setting of eps, record the acceptance rate,and draw a fi
gure showing eps
v.s. acceptance rate. What do you observe? For each setting of eps, draw the 
normalized histogram (50 bins)of collected 5K samples. What do you observe?

z(t) = position vector of ball at time t; 
r(t) = momentum vector of ball at time t;
U(z(t)) = potentional energy of ball at time t; 
K(r(t)) = kinetic energy of ball at time t;
H(z,r) = U(z) + K(r) total energy

Lecture 25: Minute 60
"""

#import numpy as np
import autograd.numpy as np
from autograd import grad
import scipy.stats as sp
import matplotlib.pyplot as plt

def pDist(z):
    return np.exp(-z**2)*((1+np.exp(-10*z-3))**-1)

def dUdz(z,e):
    return (pDist(z+e)-pDist(z))/e
        
smpls=50000; burnin=100000; total=smpls+burnin;

z = np.empty(total); z[0] = 0;
r = np.empty(total); r[0] = 0;
accept = np.zeros(total);
ratio = np.zeros(total);

unif_RV = np.random.uniform(size=total)

eps = np.array([0.005,0.01,0.1,0.2,0.5, 1]);
e = 5; eps1 = eps[e];
L = 10; # leapgrog steps 
M = np.array([1]); s = np.float(M[0]) # mass definition matrix
#accRatio = np.zeros((len(eps)))

for i in np.arange(0,total-1):
    
    r[i] = np.random.normal(0,s) 
    Kr = (1/2)*r[i]**2*(s**(-1)) # kinetic energy
    Uz = -np.log(pDist(z[i])) # potential energ
    H = Uz + Kr # total energy
    
    for steps in range(L):
                
        r_half = r[i] - (eps1/2)*dUdz(z[i], eps1);
        z_whole = z[i] + eps1*(r_half/s);
        r_whole = r_half - (eps1/2)*dUdz(z_whole, eps1);
        
    z_prime = z_whole;
    r_prime = -r_whole;
        
    Kr_prime = (1/2)*r_prime**2*(s**(-1)) # kinetic energy
    Uz_prime = -np.log(pDist(z_prime)) # potential energy
    H_prime = Uz_prime + Kr_prime # total energy
    
    ratio[i] = np.exp(-H_prime + H);
    probability = min(1., ratio[i]);
    
    if probability >= unif_RV[i]:
        z[i+1] = z_prime        
        accept[i] = 1;

    else:
        z[i+1] = z[i]

zStationary = z[burnin:];
zPicks = zStationary[0::10]

accRatio[e] = np.sum(accept[burnin:])/smpls;
print('Stepsize: ', eps[e])
print('Acceptance Ratio = %.4f' % accRatio[e])

#% plot style
plt.style.use('ggplot')
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)

#% plot chain
plt.figure(figsize=(10,4))
plt.plot(zPicks, c='turquoise', lw=1)

plt.title("Sampling Chain (eps = %.3f)" % eps[e])
plt.xlabel("Value")
plt.ylabel("Sample")

#% plot chain histogram
plt.figure(figsize=(10,4))
plt.hist(zPicks, density=True, bins=50, alpha=0.9, color='turquoise', edgecolor='white', label='HMC')
plt.plot(x0, pz(x0), c='olive', lw=1.5, label='GQ')

plt.title("Sampling Histogram (eps = %.3f)" % eps[e])
#plt.xlim(-5,5)
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

#%% plot acceptance ratio

plt.style.use('ggplot')
plt.figure(figsize=(4,4))
plt.plot(eps, accRatio, c='turquoise', lw=1.5, marker='o', markerfacecolor='k')

plt.title("Acceptance Rate")
plt.xlabel("tau")
plt.ylabel("Value")
plt.show()

