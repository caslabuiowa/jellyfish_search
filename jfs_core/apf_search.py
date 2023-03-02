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
from numba import cfunc, carray, njit
from numbalsoda import lsoda_sig, lsoda
import numpy as np
from numpy.random import default_rng
from scipy.integrate import solve_ivp

from jellyfish_search.bernstein_solvers.bernstein_least_squares import solve_least_squares
from polynomial.bernstein import Bernstein

LOG_LEVEL = logging.WARN
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
if len(logger.handlers) < 1:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LOG_LEVEL)
    logger.addHandler(stream_handler)


def generate_apf_trajectory(x0, goal, obstacles,
                            n=5, Katt_std=1, Krep_std=1, rho_std=1, d_obs=1, tf_max=60, t0=0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    Katt, Krep, rho_0 = create_perturbed_parameters(Katt_std, Krep_std, rho_std, rng)
    fn = make_lsoda_fn(goal, tuple(obstacles), d_obs, Katt, Krep, rho_0)

    # func_apf = FAPF(x0, obstacles, goal, Katt=Katt, Krep=Krep, rho_0=rho_0, d_obs=d_obs)

    # Note that max step is important otherwise the solver will return a straight line
    try:
        # res = solve_ivp(func_apf.fn, (t0, tf_max), x0, max_step=1e-1,
        #                 events=func_apf.event, dense_output=True)
        # data = (goal, tuple(obstacles), d_obs, Katt, Krep, rho_0)
        usol, success = lsoda(fn.address, x0, np.linspace(0, 60, 1001))
    except ValueError as e:
        print(f'{Katt=}\n{Krep=}\n{rho_0=}\n{t0=}\n{tf_max=}\n{x0=}')
        raise e
    # tf = res.t[-1]
    # t = np.linspace(t0, tf, 2*n)
    # sol = res.sol(t)
    # cpts = np.concatenate([[solve_least_squares(sol[0, :], n)],
    #                        [solve_least_squares(sol[1, :], n)]])

    for idx, pt in enumerate(usol):
        if np.linalg.norm(pt - goal) < 1e-3:
            break

    cpts = usol[:, :idx].T

    traj = Bernstein(cpts, t0=0, tf=60)

    return traj


def generate_piecewise_apf_trajectory(x0, goal, obstacles,
                                      n=2, m=5, Katt_std=1, Krep_std=1, rho_std=1, d_obs=1, tf_max=60, t0=0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    Katt, Krep, rho_0 = create_perturbed_parameters(Katt_std, Krep_std, rho_std, rng)

    func_apf = FAPF(x0, obstacles, goal, Katt=Katt, Krep=Krep, rho_0=rho_0, d_obs=d_obs)

    # Note that max step is important otherwise the solver will return a straight line
    res = solve_ivp(func_apf.fn, (t0, tf_max), x0, method='LSODA', max_step=1e-1,
                    events=func_apf.event, dense_output=True)
    tf = res.t[-1]
    trajs = []
    for i in range(m):
        t = np.linspace(t0 + i*(tf-t0)/m, t0 + (i+1)*(tf-t0)/m, n+1)
        sol = res.sol(t)
        cpts = np.concatenate([[sol[0, :]],
                               [sol[1, :]]])
        traj = Bernstein(cpts, t0=t[0], tf=t[-1])
        trajs.append(traj)

    return trajs


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
        return _fapf_fn(t, x, self.goal, tuple(self.obstacles), self.d_obs, self.Katt, self.Krep, self.rho_0)

    def event(self, t, x):
        distance = np.linalg.norm(x - self.goal)

        if distance < 1e-3:
            distance = 0.0

        return distance

    event.terminal = True  # Required for the integration to stop early


@njit(cache=True)
def _fapf_fn(t, x, goal, obstacles, d_obs, Katt, Krep, rho_0):
    norm = np.linalg.norm(x - goal)
    u_att_x = Katt*(x[0] - goal[0]) / norm
    u_att_y = Katt*(x[1] - goal[1]) / norm

    u_rep_x = 0
    u_rep_y = 0
    for obs in obstacles:
        rho_x = np.linalg.norm(x - obs) - d_obs
        if rho_x <= rho_0:
            # gain = (self.Krep/rho_x**3) * (1 - rho_x/self.rho_0)
            gain = -Krep*(rho_0 - rho_x) / (rho_0*rho_x**4)
            u_rep_x += gain*(x[0] - obs[0])
            u_rep_y += gain*(x[1] - obs[1])

    return np.array([-u_att_x - u_rep_x,
                     -u_att_y - u_rep_y])


def make_lsoda_fn(goal, obstacles, d_obs, Katt, Krep, rho_0):
    @cfunc(lsoda_sig)
    def _fapf_fn_lsoda(t, x_, dx, _):
        x = carray(x_, (2,))
        norm = np.linalg.norm(x - goal)
        u_att_x = Katt*(x[0] - goal[0]) / norm
        u_att_y = Katt*(x[1] - goal[1]) / norm

        u_rep_x = 0
        u_rep_y = 0
        for obs in obstacles:
            rho_x = np.linalg.norm(x - obs) - d_obs
            if rho_x <= rho_0:
                # gain = (self.Krep/rho_x**3) * (1 - rho_x/self.rho_0)
                gain = -Krep*(rho_0 - rho_x) / (rho_0*rho_x**4)
                u_rep_x += gain*(x[0] - obs[0])
                u_rep_y += gain*(x[1] - obs[1])

        dx[0] = -u_att_x - u_rep_x
        dx[1] = -u_att_y - u_rep_y

    return _fapf_fn_lsoda


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

    # fapf = FAPF(x0, obstacles, goal, Katt=10, Krep=1, rho_0=0.1, d_obs=0.5)
    # res = solve_ivp(fapf.fn, (0, 60), x0, method='LSODA', max_step=1, events=fapf.event)
    # plt.figure()
    # plt.plot(res.y[0], res.y[1])

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

    pw_trajs = []
    for i in range(100):
        print(i)
        pw_trajs.append(generate_piecewise_apf_trajectory(x0, goal, obstacles, d_obs=d_obs, rho_std=0.1,
                                                          n=3, m=10, tf_max=600, rng=rng))

    fig2, ax2 = plt.subplots()
    for pw_traj in pw_trajs:
        for traj in pw_traj:
            traj.plot(ax2, showCpts=False)

    for obs in obstacles:
        artist = Circle(obs, radius=d_obs)
        ax2.add_artist(artist)
