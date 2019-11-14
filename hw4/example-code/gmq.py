'''
This file contains sample code about how to use Gauss–Hermite quadrature to compute a specific type of integral numerically.

The general form of this type of integral is:( see https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature for more details)

F = int_{ -inf}^{+inf} e^{-x*x) f(x) dx,  (1)

in which we're calculating the integral of f(x) in the range ( -inf, +inf) weighted by e^(-x*x ).
Note that for f(x) being polynomial function, this integral is guaranteed to converge. But for some others convergence is not guaranteed.
'''

import numpy as np
import matplotlib.pyplot as plt

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

    return F

if __name__ == '__main__':

    def pz(x):
        return np.exp(-x**2)*((1+np.exp(-10*x-3))**-1)
    
    F = gauss_hermite_quad(pz, degree=5) 
    print('pz Integral =', F)
    
x0 = np.arange(-5,5,0.1)


plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,6))

font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)

#plt.plot(x0, pz(x0), c='olive', label='Normalized Dist')
plt.fill(x0, pz(x0), x0, np.zeros(len(x0)), c='olive', alpha=0.5)
plt.text(4.2, 0.7, 'Z =' + str(np.round(F,3)))
plt.xlabel('z'); plt.ylabel('Density') 
plt.title('Density Curve')
plt.legend(['Normalized Dist'])






