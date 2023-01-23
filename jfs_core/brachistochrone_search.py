#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 12:50:58 2023

@author: magicbycalvin
"""

import matplotlib.pyplot as plt
from numba import njit
import numpy as np
from scipy.optimize import minimize, Bounds, newton, toms748
import timeit

from polynomial.bernstein import Bernstein


def brachistochrone_search(x0, goal, R_std, n=5, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    goal = goal - x0
    sign_y = np.sign(goal[1])
    goal[1] *= sign_y

    traj = solve_brachistochrone_problem(goal, n, x0, sign_y, R_std, rng)

    return traj


def solve_brachistochrone_problem(goal, n, x0, sign_y, R_std, rng):
    theta_f = find_theta(goal[0], goal[1])
    R = goal[1] / (1 - np.cos(theta_f))
    R += rng.normal(scale=R_std)
    tf = theta_f * R / 9.81

    theta = np.linspace(0, theta_f, n+1)
    solution = R*np.array([theta - np.sin(theta),
                           (1 - np.cos(theta))*sign_y])
    solution += x0[:, np.newaxis]

    traj = Bernstein(solution, tf=tf)

    return traj


def find_theta(x, y):
    initial_guess = 0.5
    res = newton(lambda theta: theta_equality(x, y, theta), initial_guess, fprime=fprime)

    return res


# def find_theta2(x, y):
#     initial_guess = 0.5
#     res = newton(lambda theta: theta_equality(x, y, theta), initial_guess)

#     return res


# def find_theta3(x, y):
#     initial_guess = 0.5
#     res = newton(lambda theta: theta_equality(x, y, theta), initial_guess, fprime=fprime, fprime2=fprime2)

#     return res


# def find_thetamin(x, y):
#     initial_guess = 0.5
#     res = minimize(lambda theta: theta_equality(x, y, theta), initial_guess, bounds=Bounds(0, 2*np.pi))

#     return res


# def find_thetatoms(x, y):
#     initial_guess = 0.5
#     res = toms748(lambda theta: theta_equality(x, y, theta), 1e-3, 2*np.pi)

#     return res


@njit(cache=True)
def fprime(theta):
    return -(theta*np.sin(theta) + 2*np.cos(theta) - 2) / (theta - np.sin(theta))**2


# @njit(cache=True)
# def fprime2(theta):
#     return ((26 - 32*np.cos(theta) + 4*theta**2*np.cos(theta) + 6*np.cos(2*theta) - 12*theta*np.sin(theta) +
#              2*theta*np.sin(2*theta))
#             /
#             (4*theta**3 - 12*theta**2*np.sin(theta) + 6*theta - 6*theta*np.cos(2*theta) - 3*np.sin(theta) +
#              np.sin(3*theta)))


@njit(cache=True)
def theta_equality(x, y, theta):
    return (y/x) - (1 - np.cos(theta)) / (theta - np.sin(theta))


# def test_timings():
#     number = 100
#     repeat = 30
#     time = []
#     time2 = []
#     time3 = []
#     timemin = []
#     timetoms = []
#     for i in range(100):
#         print(f'Iteration {i}')
#         y = np.random.rand()*2*np.pi

#         time.append(timeit.repeat(lambda: find_theta(x, y), repeat=repeat, number=number))
#         time2.append(timeit.repeat(lambda: find_theta2(x, y), repeat=repeat, number=number))
#         time3.append(timeit.repeat(lambda: find_theta3(x, y), repeat=repeat, number=number))
#         timemin.append(timeit.repeat(lambda: find_thetamin(x, y), repeat=repeat, number=number))
#         timetoms.append(timeit.repeat(lambda: find_thetatoms(x, y), repeat=repeat, number=number))

#     time = np.array(time)/number
#     time2 = np.array(time2)/number
#     time3 = np.array(time3)/number
#     timemin = np.array(timemin)/number
#     timetoms = np.array(timetoms)/number

#     minimums = [x.min(axis=1) for x in [time, time2, time3, timemin, timetoms]]
#     means = [np.mean(x) for x in minimums]
#     stds = [np.std(x) for x in minimums]

#     plt.figure()
#     plt.bar(range(len(means)), means, yerr=stds)


if __name__ == '__main__':
    rng = np.random.default_rng(3)
    R_std = 0.1
    n = 7
    x0 = np.array([4, 0])
    goal = np.array([10, -3])

    # res = find_theta(x, y)

    # theta = np.linspace(0, 2*np.pi, 10001)
    # f = theta_equality(x, y, theta)

    # plt.close('all')
    # plt.plot(theta, f)
    # plt.ylim([-3, 3])

    trajs = []
    for i in range(100):
        traj = brachistochrone_search(x0, goal, R_std, n=n, rng=rng)
        trajs.append(traj)

    plt.close('all')
    fig, ax = plt.subplots()
    for traj in trajs:
        traj.plot(ax, showCpts=False)
