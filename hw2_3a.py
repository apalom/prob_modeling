# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:42:50 2019

@author: Alex Palomino
We will implement a multi-class logistic regression model for car evaluation 
task. To ensure numerical stability and avoid overfitting, we assign the feature 
weights a standard normal prior $\N(\0, \I)$.  
    
[15 points] Implement MAP estimation algorithm for multi-class logistic 
regression model. To do so, you can calculate the gradient and feed it 
to some optimization package, say, L-BFGS. Report the prediction accuracy 
on the test data.
"""

bank_test = pd.read_csv('data/hw2_bank-note/test.csv', names=['var','skew','curt','entropy','genuine'])
bank_trn = pd.read_csv('data/hw2_bank-note/train.csv', names=['var','skew','curt','entropy','genuine']);
