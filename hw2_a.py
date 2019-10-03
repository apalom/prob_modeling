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
x_std = 0.2; #data variance assumed to be known a priori
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
beta = 25; #standard deviation of data likelihood assumed to be known
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

#%%Calculate posterior distribution using the analytical solution

#generate regression matrix from input values for the line model
Phi = lineRegressor(x);

#calculate covariance matrix
SN_inv = np.linalg.inv(S0) + beta*Phi.T.dot(Phi)
SN = np.linalg.inv(SN_inv);

#calculate mean value (that is the maximum likelihood value)
mN = SN.dot(np.linalg.inv(S0).dot(m0) + beta*Phi.T.dot(t));

#generate distribution object using the stats package for plotting
PosteriorAnalytical = stats.multivariate_normal.pdf(MB,mN.ravel(),SN);

#plot the distribution
plt.figure(figsize=(4,4))
plt.contourf(M,B,PosteriorAnalytical.reshape(M.shape))
plt.plot(mbPost[0],mbPost[1],'b*',markersize=10,label='Posterior numerical'+label(mbPost));
plt.plot(mN[0],mN[1],'bx',markersize=10,label='mN: '+label(mN))
plt.title('Posterior (analytically)')
plt.xlabel('m');
plt.ylabel('b');
plt.legend(loc='lower center');

#%% Drawing models from the distribution

f = plt.figure(figsize=(12,4))
ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=1)
ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=2)

#draw parameters from distribution
N = 20;
colors = plt.cm.RdYlGn(np.linspace(0,1,N))

#draw some 50 parameters
draws = stats.multivariate_normal.rvs(mN.ravel(),SN,N)

#visualize the drawn parameters
ax1.scatter(draws[:,0],draws[:,1],c='b',s = 50,label='drawn parameters')
ax1.set_xlim(-1,1);
ax1.set_ylim(-1,1);
ax1.set_title('drawn parameters');

#draw the maximum likelihood estimate
ax1.plot(mN[0],mN[1],'r*',markersize=20,label ='mN')
ax1.legend();

#draw the line model for each parameter
tempx = np.array([[-5.0],[10]]);
for i in range(0,N):
    draw = draws[i];
    ax2.plot(tempx,linearmodel(tempx,draw),'b-',linewidth=1,alpha=0.5);#,c=colors[i],alpha=0.5);
ax2.set_title('according models');

#draw the line for the maximum likelihood parameter in red
ax2.plot(tempx,linearmodel(tempx,mN),linewidth=3,c='r');

#also show the original data points 
ax2.plot(x,t,'ko',label='data points',markeredgecolor='w');
ax2.legend();

#%% Visualizing the prediction probability

f = plt.figure(figsize=(12,4))
ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=1)
ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=2)
#f,(ax1,ax2) = plt.subplots(1,2,figsize=(8,4))

#draw 500 parameter samples from the posterior distribution 
#using the stats framework for multivariate normal distributions 
#by providingand and our calculated variance matrix 
drawsNum = 25
draws = stats.multivariate_normal.rvs(mN.ravel(),SN,drawsNum)

#plot the drawn parameters 
ax1.scatter(draws[:,0],draws[:,1],c='b',s = 20,label='drawn parameters')
ax1.plot(mN[0],mN[1],'r*',markersize=20,label ='mN')
ax1.set_xlim(-1,1);
ax1.set_ylim(-1,1);
ax1.legend();
ax1.set_title('drawn parameters');

#plot the models with a small alpha value
for draw in draws:
    tempx = np.array([[-10.0],[10.0]]);
    ax2.plot(tempx,linearmodel(tempx,draw),linewidth=20,alpha=1/100.0,color='b');

#draw the maximum likelihood estimate 
ax2.plot(tempx,linearmodel(tempx,mN),'r-',linewidth=3);
#also show the original data points
ax2.plot(x,t,'ko',label ='data points',markeredgecolor='w');
ax2.set_xlim(-5,10);
ax2.set_ylim(-5,7.5);
ax2.set_title('Visualization of Prediction Probability');

#%% Posterior Samples

# Linear Model
def linearModel(x_n,w,n):
    y_n = n*[w[0]] + x_n*w[1]
    return y_n


plt.style.use('ggplot')
plt.figure(figsize=(8,6))
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)
x=np.linspace(-1,1,20)

mb0 = np.array([[-0.3],[0.5]])

n=20;

for i in range(n):
    w_0samp = np.random.choice(Posterior.flatten(), 1)[0]
    w_1samp = np.random.choice(Posterior.flatten(), 1)[0]
    y_samp = w_0samp + x*w_1samp
    plt.plot(x,y_samp, label='[{0:.3f}, {1:.3f}]'.format(w_0samp,w_1samp))

plt.plot(x,linearModel(x,mb0,n)[0], 'k--', label='Linear Model')

plt.xlabel('x')
plt.xlim((-1,1))
plt.ylabel('y')
plt.ylim((-1,1))
plt.legend(loc='best', prop={'size': 8}, title='Weights')
plt.title('Posterior Samples')
plt.tight_layout()

#%% Prior Samples

# Linear Model
def linearModel(x_n,w,n):
    y_n = n*[w[0]] + x_n*w[1]
    return y_n


plt.style.use('ggplot')
plt.figure(figsize=(8,6))
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)
x=np.linspace(-1,1,20)

mb0 = np.array([[-0.3],[0.5]])

n=20;

for i in range(n):
    w_0samp = np.random.choice(Prior.flatten(), 1)[0]
    w_1samp = np.random.choice(Prior.flatten(), 1)[0]
    y_samp = w_0samp + x*w_1samp
    plt.plot(x,y_samp, label='[{0:.3f}, {1:.3f}]'.format(w_0samp,w_1samp))

plt.plot(x,linearModel(x,mb0,n)[0], 'k--', label='Linear Model')

plt.xlabel('x')
plt.xlim((-1,1))
plt.ylabel('y')
plt.ylim((-1,1))
plt.legend(loc='best', prop={'size': 8}, title='Weights')
plt.title('Prior Samples')
plt.tight_layout()
