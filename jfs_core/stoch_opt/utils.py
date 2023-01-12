#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 06:44:50 2022

@author: magicbycalvin
"""

from numba import njit
import numpy as np

@njit(cache=True)
def state2cpts(x, xdot, xddot, t0, tf, n):
    x = x.flatten()
    xdot = xdot.flatten()
    xddot = xddot.flatten()

    ndim = x.shape[0]
    cpts = np.empty((ndim, 3))

    cpts[:, 0] = x
    cpts[:, 1] = xdot*((tf-t0) / n) + cpts[:, 0]
    cpts[:, 2] = xddot*((tf-t0) / n)*((tf-t0) / (n-1)) + 2*cpts[:, 1] - cpts[:, 0]

    return cpts
