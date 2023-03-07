#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:13:04 2023

@author: magicbycalvin
"""
from numba import njit
import numpy as np


class FCBF:
    def __init__(self, x0, obstacles, goal, Katt=1, Krep=1, rho_0=1, d_obs=1, delta=0.01):
        self.x0 = x0
        self.obstacles = tuple(obstacles)
        self.goal = goal
        self.Katt = Katt
        self.Krep = Krep
        self.rho_0 = rho_0
        self.d_obs = d_obs
        self.delta = delta

    def fn(self, t, x):
        return _fcbf_fn(t, x, self.goal, self.obstacles, self.d_obs, self.Katt, self.Krep, self.rho_0, self.delta)

    def event(self, t, x):
        distance = np.linalg.norm(x - self.goal)

        if distance < 1e-3:
            distance = 0.0

        return distance

    event.terminal = True  # Required for the integration to stop early


def _fcbf_fn(t, x, goal, obstacles, d_obs, k_att, k_rep, rho_0, delta):
    vstar = -_grad_uatt(x, k_att, goal)

    min_dist = np.inf
    nearest_obstacle = None
    for obs in obstacles:
        dist = np.linalg.norm(x - obs)
        if dist < min_dist:
            min_dist = dist
            nearest_obstacle = obs

    if nearest_obstacle is not None:
        psi = _psi(x, k_att, k_rep, rho_0, goal, nearest_obstacle, d_obs, delta)
        if psi < 0:
            grad_h = _grad_h(x, k_rep, rho_0, nearest_obstacle, d_obs)
            vstar += -grad_h * psi/(grad_h@grad_h)

    return vstar

@njit(cache=True)
def _psi(x, k_att, k_rep, rho_0, x_goal, x_obs, d_obs, delta):
    grad_h = _grad_h(x, k_rep, rho_0, x_obs, d_obs)
    grad_uatt = _grad_uatt(x, k_att, x_goal)
    h = _h(x, k_rep, rho_0, x_obs, d_obs, delta)

    return -grad_h@grad_uatt + _alpha(h)

@njit(cache=True)
def _alpha(x):
    return x


@njit(cache=True)
def _h(x, k_rep, rho_0, x_obs, d_obs, delta):
    urep = _urep(x, k_rep, rho_0, x_obs, d_obs)

    return 1/(1 + urep) - delta


@njit(cache=True)
def _grad_h(x, k_rep, rho_0, x_obs, d_obs):
    urep = _urep(x, k_rep, rho_0, x_obs, d_obs)
    grad_urep = _grad_urep(x, k_rep, rho_0, x_obs, d_obs)

    return -grad_urep / (1 + urep)**2


@njit(cache=True)
def _uatt(x, k_att, x_goal):
    return 0.5*k_att*(x@x_goal)


@njit(cache=True)
def _grad_uatt(x, k_att, x_goal):
    ret_val = []

    for i in range(len(x)):
        ret_val.append(k_att*(x[i] - x_goal[i]))

    return np.array(ret_val)

@njit(cache=True)
def _urep(x, k_rep, rho_0, x_obs, d_obs):
    rho_x = _rho_x(x, x_obs, d_obs)

    return 0.5 * k_rep * ((1/rho_x) - (1/rho_0))**2


@njit(cache=True)
def _grad_urep(x, k_rep, rho_0, x_obs, d_obs):
    ret_val = []

    rho_x = _rho_x(x, x_obs, d_obs)
    gain = -k_rep*(rho_0 - rho_x) / (rho_0*np.linalg.norm(x - x_obs)*rho_x**3)

    for i in range(len(x)):
        ret_val.append(gain*(x[i] - x_obs[i]))

    return np.array(ret_val)


@njit(cache=True)
def _rho_x(x, x_obs, d_obs):
    return np.linalg.norm(x - x_obs) - d_obs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from scipy.integrate import solve_ivp

    goal = np.array([100, 0], dtype=float)
    x0 = np.array([0.1, 0.1], dtype=float)
    obstacles = [np.array([8.5, 0], dtype=float),  # Obstacle positions (m)
                 np.array([20, 2], dtype=float),
                 np.array([60, 1], dtype=float),
                 np.array([40, 2], dtype=float),
                 np.array([50, -3], dtype=float),
                 np.array([80, -3], dtype=float),
                 np.array([30, -1], dtype=float)]

    fcbf = FCBF(x0, obstacles, goal)
    sol = solve_ivp(fcbf.fn, (0, 100), x0, max_step=1e-2)

    plt.close('all')
    fig, ax = plt.subplots()
    ax.plot(sol.y[0, :], sol.y[1, :])
    for obs in obstacles:
        ax.add_artist(Circle(obs, radius=fcbf.d_obs))
