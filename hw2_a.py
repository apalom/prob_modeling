# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 17:09:03 2019

@author: Alex
"""

import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt


def line(x,b,m):
    return b+m*x;
def label(theta):
    return "[%2.2f, %2.2f]" %(theta[0],theta[1]);

def likelihood(t,model,x,w,std):
    mu = model(x,w);
    ps = stats.norm.pdf(t,mu,std)
    l = 1;
    for p in ps:
        l = l*p;
    return l;

def lineRegressor(x):
    x = np.array(x,ndmin=2);#make sure scalars are treated as matrices
    ones = np.ones((np.shape(x)[0],1));
    phi = np.concatenate((ones,x), axis=1);
    return phi;

def linearmodel(x,w):
    phi = lineRegressor(x);
    return phi.dot(w);

#create data from polyonimal model
N = 20;
x_std = 0.3; #data variance assumed to be known a priori
mb0 = np.array([[-0.3],[0.5]])
x = np.vstack(np.linspace(0,3,int(N)));
t = linearmodel(x,mb0) + np.vstack(np.random.normal(0,x_std,x.shape[0]))

#draw data
tempx = np.vstack(np.linspace(-5,5,10));
plt.plot(tempx,linearmodel(tempx,mb0),'k--',label=label(mb0.ravel()));
plt.plot(x,t,'k.',markersize=20,label='data points',markeredgecolor='w');
plt.xlabel('x');
plt.ylabel('y');
plt.legend();

#create array to cover parameter space
res = 100;
M,B = np.meshgrid(np.linspace(-1,1,res),np.linspace(-1,1,res));
MB = np.c_[M.ravel(),B.ravel()];

#calculate likelihood function
beta = 1; #standard deviation of data likelihood assumed to be known
L = np.array([likelihood(t,linearmodel,x,mb.reshape(2,1),beta) for mb in MB]).reshape(M.shape)

#draw
f,(ax2,ax3,ax4) = plt.subplots(1,3,figsize=(15,5),sharey=True);

#draw likelihood function
ax2.contourf(M,B,L);
ax2.set_title('Data Likelihood')
ax2.set_xlabel('m');
ax2.set_ylabel('b')
ax2.plot(mb0[0],mb0[1],'k*',markersize=10,label='True Parameter: '+label(mb0));
mbMLE = MB[L.argmax()]
ax2.plot(mbMLE[0],mbMLE[1],'r*',markersize=10,label='Max Likelihood: '+label(mbMLE));
ax2.legend(loc='lower center')

#prior distribution
S0 = np.array([[0.1,0],[0.0,0.1]]);
m0 = np.array([[-0.2],[0.7]]);
Prior = stats.multivariate_normal.pdf(MB,m0.ravel(),S0);
Prior = Prior.reshape(M.shape);
ax3.contourf(M,B,Prior)
ax3.set_title('Prior Prob. Dist.')
ax3.set_xlabel('m');
ax3.plot(mb0[0],mb0[1],'k*',markersize=10,label='True Parameter: '+label(mb0));
ax3.plot(m0[0],m0[1],'g*',markersize=10,label='Max Prior: '+label(m0));
ax3.legend(loc='lower center')

#posterior
Posterior = np.multiply(Prior,L)
ax4.contourf(M,B,Posterior)
ax4.set_title('Posterior Prob. Dist.')
ax4.set_xlabel('m');
mbPost = MB[Posterior.argmax()]
ax4.plot(mb0[0],mb0[1],'k*',markersize=10,label='True Parameter: '+label(mb0));
ax4.plot(mbMLE[0],mbMLE[1],'r*',markersize=10,label='Max Likelihood: '+label(mbMLE));
ax4.plot(m0[0],m0[1],'g*',markersize=10,label='Max Prior: '+label(m0));
ax4.plot(mbPost[0],mbPost[1],'b*',markersize=10,label='Max Posterior'+label(mbPost));
ax4.legend(loc='lower center');

#%%

mu = linearmodel(x,MB[0]);
ps = stats.norm.pdf(t,mu,x_std)
l = 1;
for p in ps:
    l = l*p;
    
#%%
    
L2 = np.array([likelihood(t,linearmodel,x,MB[2].reshape(2,1),beta))
