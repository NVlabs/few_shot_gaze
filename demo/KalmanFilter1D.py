#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Pavlo Molchanov, Shalini De Mello.
# --------------------------------------------------------

import numpy as np

class Kalman1D(object):

    def __init__(self, R=0.001**2, sz=100):
        self.Q = 1e-5 # process variance
        # allocate space for arrays
        self.xhat = np.zeros(sz, dtype=complex)      # a posteri estimate of x
        self.P = np.zeros(sz, dtype=complex)         # a posteri error estimate
        self.xhatminus = np.zeros(sz, dtype=complex) # a priori estimate of x
        self.Pminus = np.zeros(sz, dtype=complex)    # a priori error estimate
        self.K = np.zeros(sz, dtype=complex)         # gain or blending factor
        self.R = R # estimate of measurement variance, change to see effect
        self.sz = sz
        # intial guesses
        self.xhat[0] = 0.0
        self.P[0] = 1.0
        self.k = 1
    
    def update(self, val):
        k = self.k % self.sz
        km = (self.k-1) % self.sz
        self.xhatminus[k] = self.xhat[km]
        self.Pminus[k] = self.P[km] + self.Q
    
        # measurement update
        self.K[k] = self.Pminus[k]/( self.Pminus[k]+self.R )
        self.xhat[k] = self.xhatminus[k]+self.K[k]*(val-self.xhatminus[k])
        self.P[k] = (1-self.K[k])*self.Pminus[k]
        self.k = self.k + 1
        return self.xhat[k]
