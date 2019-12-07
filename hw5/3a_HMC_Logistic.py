# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:58:06 2019

@author: Alex Palomino
"""

import numpy as np
import scipy.stats as sp
import pandas as pd
import matplotlib.pyplot as plt

bank_test = pd.read_csv('bank-note/test.csv', names=['var','skew','curt','entropy','genuine']);
bank_trn = pd.read_csv('bank-note/train.csv', names=['var','skew','curt','entropy','genuine']);