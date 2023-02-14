#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:16:32 2023

@author: magicbycalvin
"""

from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from jfs_core.apf_search import FAPF, generate_apf_trajectory
from bernstein_bvp_solver import solve_bvp

from polynomial.bernstein import Bernstein


def fn_apf1(fapf, x, ydots):
    return fapf.fn(None, ydots[0](x))[0]


def fn_apf2(fapf, x, ydots):
    return fapf.fn(None, ydots[0](x))[1]


def basic_rep_fn(*args):
    return [1, 1]


def fn(x, ydots):
    return ode(x, ydots[0](x))


def ode(t, x):
    # return -10*x
    if t > 0.5:
        return -3*x
    else:
        return -2.9*x


def fn2(x, ydots):
    return ode2(x, [ydots[0](x), ydots[1](x)])[1]


def ode2(t, x):
    return [x[1],
            -10*x[0] ]#- 5*x[1]]
    # if t > 0.5:
    #     return [x[1],
    #             -9*x[0] - 5*x[1]]
    # else:
    #     return [x[1],
    #             -10*x[0] - 5*x[1]]


def basic_test():
    func = [fn]
    a = np.array([[5]], dtype=float)
    b = np.empty((0,0))
    k = 1
    l = 0
    ndim = 1
    N = 15
    m = k+l

    cpts = solve_bvp(func, k, l, N, a, b, ndim)

    c = Bernstein(cpts)
    ax = c.plot()

    res = solve_ivp(ode, (0, 1), a[0], max_step=1e-2)

    print(f'{res.y[0, -1] - c.cpts[0, -1]=}')

    ax.plot(res.t, res.y[0, :])
    ax.legend(('BP', 'cpts', 'numerical'))


def basic_test2():
    func = [fn2]
    a = np.array([[5, -1]], dtype=float)
    b = np.empty((0,0))
    k = 2
    l = 0
    ndim = 1
    N = 15
    m = k+l

    cpts = solve_bvp(func, k, l, N, a, b, ndim)

    c = Bernstein(cpts)
    ax = c.plot()

    res = solve_ivp(ode2, (0, 1), a[0], max_step=1e-2)

    print(f'{res.y[0, -1] - c.cpts[0, -1]=}')

    ax.plot(res.t, res.y[0, :])
    ax.legend(('BP', 'cpts', 'numerical'))


def basic_test3():
    func = [fn2]
    a = np.array([[5]], dtype=float)
    b = np.array([[0.36]])
    k = 1
    l = 1
    ndim = 1
    N = 15
    m = k+l

    cpts = solve_bvp(func, k, l, N, a, b, ndim)

    c = Bernstein(cpts)
    ax = c.plot()

    # res = solve_ivp(ode2, (0, 1), a[0], max_step=1e-2)

    # print(f'{res.y[0, -1] - c.cpts[0, -1]=}')

    # ax.plot(res.t, res.y[0, :])
    # ax.legend(('BP', 'cpts', 'numerical'))


if __name__ == '__main__':
    plt.close('all')

    print('=====\nTEST 1\n=====')
    basic_test()
    print('=====\nTEST 2\n=====')
    basic_test2()
    print('=====\nTEST 3\n=====')
    basic_test3()

    d_obs = 0.5
    goal = np.array([10, 3], dtype=float)
    goal2 = np.array([15, 5], dtype=float)
    x0 = np.array([0, 0], dtype=float)

    obstacles = [np.array([5, -100], dtype=float),  # Obstacle positions (m)
                 # np.array([20, 2], dtype=float),
                 # np.array([60, 1], dtype=float),
                 # np.array([40, 2], dtype=float),
                 # np.array([50, -3], dtype=float),
                 # np.array([80, -3], dtype=float),
                 # np.array([30, -1], dtype=float),
                 ]

    N = 100
    a = x0[:, np.newaxis]
    b = np.empty((0, 0))

    k = a.shape[1]
    l = b.shape[1]

    # ###################################################################################################################
    # ### METHOD 1 - Using rho_0, APF cost
    # ###################################################################################################################
    # fapf = FAPF(x0, obstacles, goal2, Katt=15.81, Krep=1, rho_0=0.1, d_obs=d_obs, d=1e-3)#, goal2=goal2)
    # f = [lambda x, ydots: fn_apf1(fapf, x, ydots),
    #      lambda x, ydots: fn_apf2(fapf, x, ydots)]
    # ndim = len(f)

    # # Bernstein BVP method
    # cpts = solve_bvp(f, k, l, N, a, b, ndim)
    # traj = Bernstein(cpts)

    # # Numerical IVP solver
    # res = solve_ivp(fapf.fn, (0, 1), x0, method='LSODA', max_step=1e-3,
    #                 events=fapf.event, dense_output=True)
    # tf = res.t[-1]
    # t = np.linspace(0, tf, 101)
    # y = res.sol(t)

    # # Plot everything
    # fig, ax = plt.subplots()
    # ax.plot(traj.cpts[0, :], traj.cpts[1, :], '.--', lw=3, ms=15)
    # ax.plot(traj.curve[0, :], traj.curve[1, :], lw=3)
    # plt.plot(y[0, :], y[1, :], '-', lw=3)

    # for obs in obstacles:
    #     artist = Circle(obs, radius=d_obs)
    #     ax.add_artist(artist)

    # plt.xlim([y[0, :].min() - 3, y[0, :].max() + 3])
    # plt.ylim([y[1, :].min() - 3, y[1, :].max() + 3])

    # ###################################################################################################################
    # ### METHOD 2 - Using rho_0, basic cost
    # ###################################################################################################################
    # fapf = FAPF(x0, obstacles, goal, Katt=10, Krep=1, rho_0=0.1, d_obs=d_obs, d=1e-3, gain_fn=basic_rep_fn)
    # f = [lambda x, ydots: fn_apf1(fapf, x, ydots),
    #      lambda x, ydots: fn_apf2(fapf, x, ydots)]
    # ndim = len(f)

    # # Bernstein BVP method
    # cpts = solve_bvp(f, k, l, N, a, b, ndim)
    # traj = Bernstein(cpts)

    # # Numerical IVP solver
    # res = solve_ivp(fapf.fn, (0, 3000), x0, method='LSODA', max_step=1e-1,
    #                 events=fapf.event, dense_output=True)
    # tf = res.t[-1]
    # t = np.linspace(0, tf, 101)
    # y = res.sol(t)

    # # Plot everything
    # fig2, ax2 = plt.subplots()
    # ax2.plot(traj.cpts[0, :], traj.cpts[1, :], '.--', lw=3, ms=15)
    # ax2.plot(traj.curve[0, :], traj.curve[1, :], lw=3)
    # plt.plot(y[0, :], y[1, :], '-', lw=3)

    # for obs in obstacles:
    #     artist = Circle(obs, radius=d_obs)
    #     ax2.add_artist(artist)

    # plt.xlim([y[0, :].min() - 3, y[0, :].max() + 3])
    # plt.ylim([y[1, :].min() - 3, y[1, :].max() + 3])

    # ###################################################################################################################
    # ### METHOD 3 - Ignoring rho_0, APF cost
    # ###################################################################################################################
    # fapf = FAPF(x0, obstacles, goal, Katt=10, Krep=1, rho_0=0.1, d_obs=d_obs, d=1e-3, disable_rho0=True)
    # f = [lambda x, ydots: fn_apf1(fapf, x, ydots),
    #      lambda x, ydots: fn_apf2(fapf, x, ydots)]
    # ndim = len(f)

    # # Bernstein BVP method
    # cpts = solve_bvp(f, k, l, N, a, b, ndim)
    # traj = Bernstein(cpts)

    # # Numerical IVP solver
    # res = solve_ivp(fapf.fn, (0, 3000), x0, method='LSODA', max_step=1e-1,
    #                 events=fapf.event, dense_output=True)
    # tf = res.t[-1]
    # t = np.linspace(0, tf, 101)
    # y = res.sol(t)

    # # Plot everything
    # fig3, ax3 = plt.subplots()
    # ax3.plot(traj.cpts[0, :], traj.cpts[1, :], '.--', lw=3, ms=15)
    # ax3.plot(traj.curve[0, :], traj.curve[1, :], lw=3)
    # plt.plot(y[0, :], y[1, :], '-', lw=3)

    # for obs in obstacles:
    #     artist = Circle(obs, radius=d_obs)
    #     ax3.add_artist(artist)

    # plt.xlim([y[0, :].min() - 3, y[0, :].max() + 3])
    # plt.ylim([y[1, :].min() - 3, y[1, :].max() + 3])

    # ###################################################################################################################
    # ### METHOD 4 - Ignoring rho_0, Basic cost
    # ###################################################################################################################
    # fapf = FAPF(x0, obstacles, goal, Katt=10, Krep=1, rho_0=0.1, d_obs=d_obs, d=1e-3, disable_rho0=True,
    #             gain_fn=basic_rep_fn)
    # f = [lambda x, ydots: fn_apf1(fapf, x, ydots),
    #      lambda x, ydots: fn_apf2(fapf, x, ydots)]
    # ndim = len(f)

    # # Bernstein BVP method
    # cpts = solve_bvp(f, k, l, N, a, b, ndim)
    # traj = Bernstein(cpts)

    # # Numerical IVP solver
    # res = solve_ivp(fapf.fn, (0, 3000), x0, method='LSODA', max_step=1e-1,
    #                 events=fapf.event, dense_output=True)
    # tf = res.t[-1]
    # t = np.linspace(0, tf, 101)
    # y = res.sol(t)

    # # Plot everything
    # fig4, ax4 = plt.subplots()
    # ax4.plot(traj.cpts[0, :], traj.cpts[1, :], '.--', lw=3, ms=15)
    # ax4.plot(traj.curve[0, :], traj.curve[1, :], lw=3)
    # plt.plot(y[0, :], y[1, :], '-', lw=3)

    # for obs in obstacles:
    #     artist = Circle(obs, radius=d_obs)
    #     ax4.add_artist(artist)

    # plt.xlim([y[0, :].min() - 3, y[0, :].max() + 3])
    # plt.ylim([y[1, :].min() - 3, y[1, :].max() + 3])
