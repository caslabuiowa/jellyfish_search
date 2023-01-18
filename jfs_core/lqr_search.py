#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:31:49 2022

@author: magicbycalvin
"""

import logging

from control import lqr
import matplotlib.pyplot as plt
from numba import cfunc, carray
from numbalsoda import lsoda_sig, lsoda
import numpy as np
from numpy.random import default_rng

from polynomial.bernstein import Bernstein

LOG_LEVEL = logging.WARN
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
if len(logger.handlers) < 1:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LOG_LEVEL)
    logger.addHandler(stream_handler)


def generate_lqr_trajectory(x0, goal, Q_std, R_std, x0_std, tf, n=5, rng=None):
    if rng is None:
        rng = default_rng()

    logger.debug('Initializing LQR problem')
    A, B, Q, R = initialize_lqr_problem(Q_std, R_std, rng)
    logger.debug('Perturbing initial state')
    x0_perturbed = perturb_initial_state(x0, goal, x0_std, rng)

    logger.debug((f'{A=}\n'
                  f'{B=}\n'
                  f'{Q=}\n'
                  f'{R=}\n'
                  f'{x0=}\n'
                  f'{x0_perturbed=}'))

    traj = solve_lqr_problem(A, B, Q, R, x0_perturbed, n, tf, goal)

    return traj


def initialize_lqr_problem(Q_std, R_std, rng):
    A = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
    B = np.array([[0, 0],
                  [0, 0],
                  [0, 0],
                  [1, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 1]], dtype=float)

    Q = rng.normal(0, Q_std, (8, 8))
    R = rng.normal(0, R_std, (2, 2))
    Q = (0.5*Q.T@Q).round(6)
    R = (0.5*R.T@R).round(6)

    return A, B, Q, R


def perturb_initial_state(x0, goal, std, rng):
    x0_perturbed = x0.copy()
    # Subtract goal because the LQR controller tries to bring all states back to zero
    # Only perturb the position since we want to keep the higher order derivatives of the initial state constant
    # x0_perturbed += rng.normal([-goal[0], 0, 0, 0,
    #                             -goal[1], 0, 0, 0],
    #                            std)
    x0_perturbed[0] -= rng.normal(goal[0], std)
    x0_perturbed[4] -= rng.normal(goal[1])

    return x0_perturbed


def solve_lqr_problem(A, B, Q, R, x0, n, tf, goal):
    K, S, E = lqr(A, B, Q, R)

    usol, success = lsoda(fn.address, x0, np.linspace(0, tf, n+1), data=(A - B@K))

    logger.debug(f'{usol=}')
    cpts = np.concatenate([[usol[:, 0]],
                           [usol[:, 4]]], axis=0)
    cpts -= cpts[:, 0, np.newaxis]
    logger.debug(f'{cpts=}')
    traj = Bernstein(cpts, tf=tf)

    return traj


@cfunc(lsoda_sig)
def fn(t, u, du, p):
    u_ = carray(u, (8,))
    p_ = carray(p, (8, 8))
    tmp = p_@u_
    for i in range(8):
        du[i] = tmp[i]


if __name__ == '__main__':
    seed = 3
    rng = default_rng(seed)

    n = 7
    t0 = 0
    tf = 10
    dt = 1
    ndim = 2
    t_max = 0.95
    Q_std = 1 #10
    R_std = 1 #300
    x0_std = 1

    safe_planning_radius = 10
    safe_dist = 2
    vmax = 3
    wmax = np.pi/4

    obstacles = [np.array([8, 0], dtype=float),  # Obstacle positions (m)
                 np.array([20, 2], dtype=float),
                 np.array([60, 1], dtype=float),
                 np.array([40, 2], dtype=float),
                 np.array([50, -3], dtype=float),
                 np.array([80, -3], dtype=float),
                 np.array([30, -1], dtype=float)]

    goal = np.array([100, 0], dtype=float)
    initial_position = np.array([0, 0], dtype=float)
    initial_velocity = np.array([0, 0], dtype=float)
    initial_acceleration = np.array([0, 0], dtype=float)
    x0 = np.concatenate([[initial_position],
                         [initial_velocity],
                         [initial_acceleration],
                         [[0, 0]]], axis=0).T.reshape(-1)

    # logger.debug('debug')
    # logger.info('info')
    # logger.warning('warning')
    # logger.error('error')
    # logger.critical('critical')

    traj = generate_lqr_trajectory(x0, goal, Q_std, R_std, x0_std, tf, n=n, rng=rng)
    plt.close('all')
    traj.plot()
