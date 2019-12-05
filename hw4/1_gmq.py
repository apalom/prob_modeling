'''
This file contains sample code about how to use Gauss–Hermite quadrature to compute a specific type of integral numerically.

The general form of this type of integral is:( see https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature for more details)

F = int_{ -inf}^{+inf} e^{-x*x) f(x) dx,  (1)

in which we're calculating the integral of f(x) in the range ( -inf, +inf) weighted by e^(-x*x ).
Note that for f(x) being polynomial function, this integral is guaranteed to converge. But for some others convergence is not guaranteed.
'''

import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import seaborn as sns

def gauss_hermite_quad(f, degree):
    '''
    Calculate the integral (1) numerically.
    :param f: target function, takes a array as input x = [x0, x1,...,xn], and return a array of function values f(x) = [f(x0),f(x1), ..., f(xn)]
    :param degree: integer, >=1, number of points
    :return:
    '''

    points, weights = np.polynomial.hermite.hermgauss(degree)

    #function values at given points
    f_x = f(points)

    #weighted sum of function values
    F = np.sum(f_x  * weights)
    #print('weights = ', weights)
    
    return F

if __name__ == '__main__':

    def pz(x):
        return np.exp(-x**2)*((1+np.exp(-10*x-3))**-1)
    
    def dpz(x):
        return -2*np.exp(-x**2 + 10*x + 3)*(x*np.exp(10*x + 3)+x-5)*(np.exp(10*x + 3)+1)**-2
    
    def logpz(x):
        return np.log(np.exp(-x**2)*((1+np.exp(-10*x-3))**-1))
    
    Z = gauss_hermite_quad(pz, degree=5) 
    z0 = gauss_hermite_quad(dpz, degree=5)
    A = -1 * gauss_hermite_quad(logpz, degree=5)
    print('pz Integral =', Z)
    

#%% Part A    
x0 = np.arange(-5,5,0.1)

plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,6))

font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)

#plt.plot(x0, pz(x0), c='olive', label='Normalized Dist')
#plt.plot(x0, pz(x0), c='olive', alpha=0.3)
plt.fill(x0, pz(x0), x0, np.zeros(len(x0)), c='olive', alpha=0.3)
plt.axvline(z0, color='k', linestyle=':', lw=1.5)
plt.text(4.2, 0.72, 'Z =' + str(np.round(Z,3)))
plt.text(4.2, 0.62, 'z0 =' + str(np.round(z0,3)))
plt.xlabel('z'); plt.ylabel('Density') 
plt.title('Density Curve')
plt.legend(['z0','Normalized Dist'])

#%% Part B - Laplace Approximation

def gauss(x, mu, sigma):
    return np.exp(-((x-mu)**2.) / (2*(sigma**2 )))

A_inv = A**(-1)    
yL = gauss(x0, z0, A_inv)

plt.figure(figsize=(12,6))

plt.fill(x0, pz(x0), x0, np.zeros(len(x0)), c='olive', alpha=0.3)
plt.axvline(z0, color='k', linestyle=':', lw=2.25)
plt.plot(x0, yL, lw=1.5)
plt.text(4.2, 0.72, 'Z =' + str(np.round(Z,3)))
plt.text(4.2, 0.62, 'z0 =' + str(np.round(z0,3)))
plt.xlabel('z'); plt.ylabel('Density') 
plt.title('Density Curves')
plt.legend(['z0','Laplce Approximation','Normalized Dist'])

#%% Part C - Variational Inference 
# .... Look at section 10.6 of textbook

m0 = z0; s0 = A_inv; S0_inv = A;

Sn_inv = S0_inv+np.sum(pz(x0))
Sn = Sn_inv**-1
mn = Sn*(S0_inv*m0+np.dot(pz(x0),x0))

yV = gauss(x0, mn, Sn)

plt.figure(figsize=(12,6))

plt.fill(x0, pz(x0), x0, np.zeros(len(x0)), c='olive', alpha=0.3)
plt.axvline(z0, color='k', linestyle=':', lw=2.25)
plt.plot(x0, yL, lw=1.5)
plt.plot(x0, yV, lw=1.5, color='green')
plt.text(4.2, 0.72, 'Z =' + str(np.round(Z,3)))
plt.text(4.2, 0.62, 'z0 =' + str(np.round(z0,3)))
plt.xlabel('z'); plt.ylabel('Density') 
plt.title('Density Curves')
plt.legend(['z0','Laplace Approximation','Variational Inference','Normalized Dist'])