# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:21:21 2019

@author: Alex
"""

import scipy as sp
import scipy.stats as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#% Q2 Binary Logistic & Probit Regression for Bank Note Auth

bank_test = pd.read_csv('data/hw2_bank-note/test.csv', names=['var','skew','curt','entropy','genuine'])
bank_trn = pd.read_csv('data/hw2_bank-note/train.csv', names=['var','skew','curt','entropy','genuine']);