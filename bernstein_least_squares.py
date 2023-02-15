#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:36:41 2023

@author: magicbycalvin
"""

from numba import njit
import numpy as np
from scipy.linalg import qr, solve
from scipy.special import binom


def solve_least_squares(f_samples, n, compute_residuals=False):
    l = f_samples.size

    A = create_A(n, l, f_samples)
    Q, R = qr(A)
    d = Q.T@f_samples
    d1 = d[:n+1]
    d2 = d[n+1:]

    cpts = solve(R[:n+1, :], d1)

    if compute_residuals:
        residuals = Q@np.concatenate(([0]*(n+1), d2))
        return cpts, residuals
    else:
        return cpts


def create_A(n, l, x):
    binomial_coefficients = np.array([[binom(n, i) for i in range(n+1)]]).repeat(l, axis=0)

    temp_coefficients = np.empty((l, n+1))
    for row in range(l):
        temp_coefficients[row, :] = np.array([(x[row]**i)*(1-x[row])**(n-i) for i in range(n+1)])

    A = binomial_coefficients*temp_coefficients

    return A


if __name__ == '__main__':
    n = 5
    l = 10
    x = np.linspace(0, 1, l)
    y_samp = 5*x**2

    A = create_A(n, l, x)
    print(A)

    Q, R = qr(A)
    d = Q.T@y_samp
    d1 = d[:n+1]
    d2 = d[n+1:]
