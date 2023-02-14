#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 11:11:20 2023

@author: magicbycalvin
"""
import sys
sys.path.append('/home/magicbycalvin/Projects/last_minute_comprehensive/BeBOT')

from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
from scipy.integrate import quad, solve_ivp, romb, trapezoid
from scipy.linalg import solve
from scipy.special import factorial, binom

from jfs_core.apf_search import FAPF
from polynomial.bernstein import Bernstein


def solve_bvp(f, k, l, N, a, b, ndim, return_all_trajectories=False):
    """
    Solve a boundary value problem using the iterative dual Bernstein polynomial method [1]

    Approimate the solution to a boundary value problem (BVP) using the iterative method mentioned in [1]. The BVP is
    expected to be of the following form,
        y^(m)(x) = f( x, y(x), y'(x), y''(x), ..., y^(m-1)(x) ),    (0 <= x <=1)
    with the boundary conditions,
        y^(i)(0) = a_i,    (i = 0, 1, ..., k-1)
        y^(j)(1) = b_j,    (j = 0, 1, ..., l-1)

    Note: use the keyboard interrupt (ctrl+c) to stop the iterations early and return the current trajectory.

    [1] Gospodarczyk, Przemysław, and Paweł Woźny. "An iterative approximate method of solving boundary value problems
        using dual Bernstein polynomials." arXiv preprint arXiv:1709.02162 (2017).

    Parameters
    ----------
    f : function
        Right hand side of the differential equation to be approximated of the form
            y^(m)(x) = f( x, y(x), y'(x), y''(x), ..., y^(m-1)(x) )
    k : int
        Highest order derivative of the boundary conditions at x = 0, plus one (e.g. position only -> k = 1).
    l : int
        Highest order derivative of the boundary conditions at x = 1, plus one (e.g. position only -> k = 1).
    N : int
        Degree of the approximate polynomial solution.
    a : numpy.ndarray
        Nxk matrix containing the boundary values of the function at x = 0 where each row corresponds to a new
        dimension (e.g., x, y, z) and each column corresponds to a higher order derivative (e.g., position, velocity,
        acceleration).
    b : numpy.ndarray
        Nxl matrix containing the boundary values of the function at x = 1 where each row corresponds to a new
        dimension (e.g., x, y, z) and each column corresponds to a higher order derivative (e.g., position, velocity,
        acceleration).
    ndim : int
        Number of dimensions in the problem, e.g., ndim = 2 for a 2D problem.
    return_all_trajectories: bool, optional
        Returns trajectories of orders m-1 to N rather than just the final trajectory of order N. The default is False.

    Returns
    -------
    ndimx(N+1) numpy.ndarray containing the control points of the Bernstein polynomial approximating the solution where
    each row corresponds to each dimension and each column corresponds to each control point/Bernstein coefficient.

    """
    m = k + l
    # Algorithm 3.1 Step I from [1]
    cpts = compute_outer_coefficients(m-1, ndim, k, l, a, b)

    if return_all_trajectories:
        cpts_list = [cpts]

    # Algorithm 3.1 Step II from [1]
    for n in range(m, N+1):
        try:
            print(f'{n=}')
            # Eqns 2.7 and 2.8 from [1]
            cpts_new = compute_outer_coefficients(n, ndim, k, l, a, b)
            # Lemma 2.4 from [1]
            dual_coefficients = compute_dual_coefficients(n-m)

            y = Bernstein(cpts)
            v = compute_v(ndim, f, y, n, m, k, l, dual_coefficients, cpts_new)
            G = compute_G(n, m, k, l)

            inner_cpts = solve(G, v)
            cpts_new[:, k:n-l+1] = inner_cpts.T
            cpts = cpts_new.copy()

            if return_all_trajectories:
                cpts_list.append(cpts)

            if np.any(cpts > 1e3):
                print('[!] Warning, one or more control points are greater than 1e3. Breaking.')
                break
        except KeyboardInterrupt:
            break

    if return_all_trajectories:
        return cpts_list
    else:
        return cpts


def compute_outer_coefficients(n, ndim, k, l, a, b):
    cpts = np.empty((ndim, n+1))

    # Zeroth derivative (Bernstein polynomial end point property)
    if k > 0:
        cpts[:, 0] = a[:ndim, 0]
    if l > 0:
        cpts[:, -1] = b[:ndim, 0]

    # Higher order derivatives (derivative property of Bernstein polynomials)
    for i in range(1, k):
        cpts[:, i] = (factorial(n - i) / factorial(n) * a[:ndim, i] -
                      sum([(-1)**(i - h) * binom(i, h) * cpts[:, h] for h in range(i)]))
    for j in range(1, l):
        cpts[:, n-j] = ((-1)**j * factorial(n - j) / factorial(n) * b[:ndim, j] -
                        sum([((-1)**h) * binom(j, h) * cpts[:, n-j+h] for h in range(1, j+1)]))

    return cpts


def compute_dual_coefficients(n):
    # Lemma 2.4 in [1]
    coefficients = np.empty((n+1, n+1))
    coefficients[0, :] = initial_coefficients(n)

    for i in range(n):
        for j in range(n+1):
            a = 2*(i - j)*(i + j - n)*coefficients[i, j]
            if j - 1 < 0:
                b = 0
            else:
                b = _B(j, n)*coefficients[i, j-1]
            if j + 1 > n:
                c = 0
            else:
                c = _A(j, n)*coefficients[i, j+1]
            if i - 1 < 0:
                d = 0
            else:
                d = _B(i, n)*coefficients[i-1, j]

            coefficients[i+1, j] = (1/_A(i, n))*(a + b + c - d)

    return coefficients


def initial_coefficients(n):
    ini_coeffs = np.empty(n+1)
    for j in range(n+1):
        ini_coeffs[j] = (-1)**j * (n+1) * _pochhammer(n+1-j, j+1) / factorial(j+1)

    return ini_coeffs


@njit(cache=True)
def _A(u, n):
    return (u - n)*(u + 1)


@njit(cache=True)
def _B(u, n):
    return u*(u - n - 1)


@njit(cache=True)
def _pochhammer(x, n):
    """
    Compute the Pochhammer symbol for (x)_n.

    The Pochhammer symbol is defined as Gamma(x + n) / Gamma(x) = x(x+1)...(x+n-1). For more information, see
    https://mathworld.wolfram.com/PochhammerSymbol.html

    Parameters
    ----------
    x : float
        Value to compute the Pochhammer symbol for.
    n : int
        Degree of the Pchhammer symbol to compute.

    Returns
    -------
    result : float
        Computed Pochhammer symbol.

    """
    result = 1.0
    for i in range(n):
        result *= x + i

    return result


# def forward_difference_recurrence(cpts, n, m):
#     result = np.empty(m-1)
#     result[0] =
#     p = np.empty((m, n-m))


def compute_v(ndim, f, y, n, m, k, l, dual_cpts, cpts):
    v = np.empty((n-m+1, ndim))

    for i in range(n-m+1):
        a = np.zeros((1, ndim))
        for q in range(n-m+1):
            I = compute_inner_product(ndim, f, y, q, n, m)
            a += dual_cpts[i, q]*I
        a *= factorial(n-m) / factorial(n)

        b = np.zeros((1, ndim))
        for h in range(k-i):
            b += (-1)**(m-h)*binom(m, h)*cpts[:, i+h]

        for h in range(n-l-i+1, m+1):
            b += (-1)**(m-h)*binom(m, h)*cpts[:, i+h]

        # print(f'{v=}\n{(a-b)=}')
        v[i, :] = a - b
        # print(f'{v[i, :]=}')
        # print(f'{v=}')
        # input()

    return v


def compute_G(n, m, k, l):
    G = np.empty((n-m+1, n-m+1))
    for i in range(n-m+1):
        for j in range(n-m+1):
            G[i, j] = (-1)**(l+i-j)*binom(m, j+k-i)

    return G


def compute_inner_product(ndim, f, y, q, n, m):
    ydots = [y]
    for i in range(m-1):
        ydots.append(ydots[i].diff())

    I = np.empty((1, ndim))
    for i in range(ndim):
        ### Gaussian quadrature integration
        res = quad(lambda x: f[i](x, ydots)*bernstein_basis(x, n-m, q), 0, 1, limit=1000)
        # print(f'===\n{res=}\n---\n{f[i](0.5, ydots)=}\n---\n{bernstein_basis(0.5, n-m, q)=}\n===')
        I[:, i] = res[0]

        ### Romberg integration
        # nevals = 2**9 + 1
        # # t = np.linspace(0, 1, nevals)
        # y = np.array([f[i](t, ydots)*bernstein_basis(t, n-m, q) for t in np.linspace(0, 1, nevals)])
        # I[:, i] = romb(y.squeeze(), dx=1/nevals)

        ### Trapezoidal integration
        # nevals = 100
        # y = np.array([f[i](t, ydots)*bernstein_basis(t, n-m, q) for t in np.linspace(0, 1, nevals)])
        # I[:, i] = trapezoid(y.squeeze(), dx=1/nevals)

    return I


def bernstein_basis(x, n, i):
    return binom(n, i) * x**i * (1 - x)**(n-i)


# def test_func(*ydots):
#     def fn1(x):
#         ydots[1](x)**2 + 1

#     return [fn1]


# def fn1(x, *ydots):
#     return ydots[1](x)[0, :]**2 + 1


# def fn2(x, *ydots):
#     return ydots[1](x)[1, :]**2 + 1


def fn1(x, ydots):
    return np.cos(ydots[0](x)[2, :])*10


def fn2(x, ydots):
    return np.sin(ydots[0](x)[2, :])*10


def fn3(x, ydots):
    return 0.1*10*np.pi*2


def test_fn(t, x):
    return np.array([np.cos(x[2]),
                     np.sin(x[2]),
                     0.1*np.pi*2])


def fn_apf1(fapf, x, ydots):
    return fapf.fn(None, ydots[0](x))[0]


def fn_apf2(fapf, x, ydots):
    return fapf.fn(None, ydots[0](x))[1]


if __name__ == '__main__':
    plt.close('all')

    # APF stuff
    # seed = 3
    # rng = np.random.default_rng(seed)

    d_obs = 0.5
    goal = np.array([10, 3], dtype=float)
    x0 = np.array([0, -1], dtype=float)
    obstacles = [np.array([1, 2], dtype=float),  # Obstacle positions (m)
                 np.array([2.5, 3], dtype=float)
                 ]
    obstacles = [np.array([5, 0.9], dtype=float),  # Obstacle positions (m)
                 # np.array([20, 2], dtype=float),
                 # np.array([60, 1], dtype=float),
                 # np.array([40, 2], dtype=float),
                 # np.array([50, -3], dtype=float),
                 # np.array([80, -3], dtype=float),
                 # np.array([30, -1], dtype=float),
                 ]
    fapf = FAPF(x0, obstacles, goal, Katt=11, Krep=0.01, rho_0=20, d_obs=d_obs, d=1e-3)
    # End APF stuff

    N = 100
    f = [lambda x, ydots: fn_apf1(fapf, x, ydots),
         lambda x, ydots: fn_apf2(fapf, x, ydots)]
    a = x0[:, np.newaxis]
    # b = goal[:, np.newaxis]
    # b = np.array([[0],
    #               [0]], dtype=float)
    b = np.empty((0, 0))

    # a = np.array([
    #     [1, 10, 2],
    #     [2, 1, 3]], dtype=float)
    # b = np.array([
    #     [8, 1, -0.1],
    #     [10, 2, -0.2]], dtype=float)

    k = a.shape[1]
    l = b.shape[1]
    ndim = len(f)

    cpts = solve_bvp(f, k, l, N, a, b, ndim)
    print(f'{cpts}')

    traj = Bernstein(cpts)
    # print(f'{traj}')
    # print(f'{traj.diff()}')
    # print(f'{traj.diff().diff()}')

    # x = np.linspace(0, 1, 1001)
    # y = -np.log(np.cos(x-0.5)/np.cos(0.5))

    # traj.plot()
    # plt.plot(x, y)

    # res = solve_ivp(test_fn, (0, 10), a.squeeze(), t_eval=np.linspace(0, 10, 1001))

    fig, ax = plt.subplots()
    ax.plot(traj.cpts[0, :], traj.cpts[1, :], '.--', lw=3, ms=15)
    ax.plot(traj.curve[0, :], traj.curve[1, :], lw=3)
    # plt.plot(res.y[0, :], res.y[1, :], '--', lw=3)

    for obs in obstacles:
        artist = Circle(obs, radius=d_obs)
        ax.add_artist(artist)
