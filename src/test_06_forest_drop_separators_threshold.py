# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 11:38:35 2021

@author: XM78AT
"""

import numpy as np
import matplotlib.pyplot as plt 


import mfcf as mfcf
import gain_functions as gf

p = 15
T = 100

np.random.seed(seed=1525)
X = np.random.normal(0,1,(T,p))

C = np.corrcoef(X, rowvar=False)

ctl = mfcf.mfcf_control()
ctl['threshold'] = 0.01
ctl['drop_sep'] = True
ctl['min_clique_size'] = 1
ctl['max_clique_size'] = 2
gain_function = gf.sumsquares_gen

cliques, separators, peo, gt = mfcf.mfcf(C, ctl, gain_function)
