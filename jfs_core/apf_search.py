#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:05:18 2023

@author: magicbycalvin
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:31:49 2022

@author: magicbycalvin
"""

import sys
sys.path.append('..')

import logging

from control import lqr
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from numba import cfunc, carray
from numbalsoda import lsoda_sig, lsoda
import numpy as np
from numpy.random import default_rng
from scipy.integrate import solve_ivp

from bernstein_solvers.bernstein_least_squares import solve_least_squares
from polynomial.bernstein import Bernstein

LOG_LEVEL = logging.WARN
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
if len(logger.handlers) < 1:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LOG_LEVEL)
    logger.addHandler(stream_handler)


# def generate_lqr_trajectory(x0, goal, Q_std, R_std, x0_std, tf, n=5, rng=None):
#     if rng is None:
#         rng = default_rng()

#     logger.debug('Initializing LQR problem')
#     A, B, Q, R = initialize_lqr_problem(Q_std, R_std, rng)
#     logger.debug('Perturbing initial state')
#     x0_perturbed = perturb_initial_state(x0, goal, x0_std, rng)

#     logger.debug((f'{A=}\n'
#                   f'{B=}\n'
#                   f'{Q=}\n'
#                   f'{R=}\n'
#                   f'{x0=}\n'
#                   f'{x0_perturbed=}'))

#     traj = solve_lqr_problem(A, B, Q, R, x0_perturbed, n, tf, goal)

#     return traj


# def initialize_lqr_problem(Q_std, R_std, rng):
#     A = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 1, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 1, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 1, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 1, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 1],
#                   [0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
#     B = np.array([[0, 0],
#                   [0, 0],
#                   [0, 0],
#                   [1, 0],
#                   [0, 0],
#                   [0, 0],
#                   [0, 0],
#                   [0, 1]], dtype=float)

#     Q = rng.normal(0, Q_std, (8, 8))
#     R = rng.normal(0, R_std, (2, 2))
#     Q = (0.5*Q.T@Q).round(6)
#     R = (0.5*R.T@R).round(6)

#     return A, B, Q, R


# def perturb_initial_state(x0, goal, std, rng):
#     x0_perturbed = x0.copy()
#     # Subtract goal because the LQR controller tries to bring all states back to zero
#     # Only perturb the position since we want to keep the higher order derivatives of the initial state constant
#     # x0_perturbed += rng.normal([-goal[0], 0, 0, 0,
#     #                             -goal[1], 0, 0, 0],
#     #                            std)
#     x0_perturbed[0] -= rng.normal(goal[0], std)
#     x0_perturbed[4] -= rng.normal(goal[1])

#     return x0_perturbed


# def solve_lqr_problem(A, B, Q, R, x0, n, tf, goal):
#     K, S, E = lqr(A, B, Q, R)

#     usol, success = lsoda(fn.address, x0, np.linspace(0, tf, n+1), data=(A - B@K))

#     logger.debug(f'{usol=}')
#     cpts = np.concatenate([[usol[:, 0]],
#                            [usol[:, 4]]], axis=0)
#     cpts -= cpts[:, 0, np.newaxis]
#     logger.debug(f'{cpts=}')
#     traj = Bernstein(cpts, tf=tf)

#     return traj


# @cfunc(lsoda_sig)
# def fn(t, u, du, p):
#     u_ = carray(u, (8,))
#     p_ = carray(p, (8, 8))
#     tmp = p_@u_
#     for i in range(8):
#         du[i] = tmp[i]


def generate_apf_trajectory(x0, goal, obstacles,
                            n=5, Katt_std=1, Krep_std=1, rho_std=1, d_obs=1, tf_max=60, t0=0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    Katt, Krep, rho_0 = create_perturbed_parameters(Katt_std, Krep_std, rho_std, rng)

    func_apf = FAPF(x0, obstacles, goal, Katt=Katt, Krep=Krep, rho_0=rho_0, d_obs=d_obs)

    # Note that max step is important otherwise the solver will return a straight line
    res = solve_ivp(func_apf.fn, (t0, tf_max), x0, method='LSODA', max_step=1e-1,
                    events=func_apf.event, dense_output=True)
    tf = res.t[-1]
    t = np.linspace(t0, tf, 2*n)
    # cpts = res.sol(t)
    sol = res.sol(t)
    cpts = np.concatenate([[solve_least_squares(sol[0, :], n)],
                           [solve_least_squares(sol[1, :], n)]])

    traj = Bernstein(cpts, t0=t0, tf=tf)

    return traj

def create_perturbed_parameters(Katt_std, Krep_std, rho_std, rng):
    Katt = rng.lognormal(mean=1, sigma=Katt_std)
    Krep = rng.lognormal(mean=1, sigma=Krep_std)
    rho_0 = rng.lognormal(mean=0.5, sigma=rho_std)

    return Katt, Krep, rho_0


class FAPF:
    def __init__(self, x0, obstacles, goal, Katt=1, Krep=1, rho_0=1, d_obs=1, t0=0):
        self.x0 = x0
        self.obstacles = obstacles
        self.goal = goal
        self.Katt = Katt
        self.Krep = Krep
        self.rho_0 = rho_0
        self.d_obs = d_obs
        self.t0 = t0

    def fn(self, t, x):
        norm = np.linalg.norm(x - self.goal)
        # print(f'{norm=}')
        u_att_x = self.Katt*(x[0] - self.goal[0]) / norm
        u_att_y = self.Katt*(x[1] - self.goal[1]) / norm

        u_rep_x = 0
        u_rep_y = 0
        for obs in self.obstacles:
            rho_x = np.linalg.norm(x - obs) - self.d_obs
            if rho_x <= self.rho_0:
                # gain = (self.Krep/rho_x**3) * (1 - rho_x/self.rho_0)
                gain = -self.Krep*(self.rho_0 - rho_x) / (self.rho_0*rho_x**4)
                u_rep_x += gain*(x[0] - obs[0])
                u_rep_y += gain*(x[1] - obs[1])

        return np.array([-u_att_x - u_rep_x,
                         -u_att_y - u_rep_y])

    def event(self, t, x):
        distance = np.linalg.norm(x - self.goal)

        if distance < 1e-3:
            distance = 0.0

        return distance

    event.terminal = True  # Required for the integration to stop early


if __name__ == '__main__':
    plt.close('all')
    seed = 3
    rng = default_rng(seed)

    n = 10
    d_obs = 0.5
    goal = np.array([100, 0], dtype=float)
    x0 = np.array([0.1, 0.1], dtype=float)
    obstacles = [np.array([1, 2], dtype=float),  # Obstacle positions (m)
                 np.array([2.5, 3], dtype=float)]
    obstacles = [np.array([8, 0], dtype=float),  # Obstacle positions (m)
                  np.array([20, 2], dtype=float),
                  np.array([60, 1], dtype=float),
                  np.array([40, 2], dtype=float),
                  np.array([50, -3], dtype=float),
                  np.array([80, -3], dtype=float),
                  np.array([30, -1], dtype=float)]

    fapf = FAPF(x0, obstacles, goal, Katt=10, Krep=1, rho_0=0.1, d_obs=0.5)
    res = solve_ivp(fapf.fn, (0, 60), x0, method='LSODA', max_step=1, events=fapf.event)
    plt.figure()
    plt.plot(res.y[0], res.y[1])

    # traj = generate_apf_trajectory(x0, goal, obstacles, n=n)
    # traj.plot(showCpts=True)

    trajs = []
    for i in range(100):
        print(i)
        trajs.append(generate_apf_trajectory(x0, goal, obstacles, n=n, d_obs=d_obs, rho_std=0.1, tf_max=600, rng=rng))

    fig, ax = plt.subplots()
    for traj in trajs:
        traj.plot(ax, showCpts=False)

    for obs in obstacles:
        artist = Circle(obs, radius=d_obs)
        ax.add_artist(artist)
