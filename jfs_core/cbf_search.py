#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:13:04 2023

@author: magicbycalvin
"""
from numbalsoda import lsoda, lsoda_sig
from numba import carray, cfunc, njit
from numba.experimental import jitclass
import numpy as np
from scipy.integrate import solve_ivp

from polynomial.bernstein import Bernstein


# def generate_cbf_trajectory(x0, goal, obstacles,
#                             n=5, Katt_std=1, Krep_std=1, rho_std=1, d_obs=1, tf_max=60, t0=0, delta=0.01, rng=None,
#                             debug=False):
#     if rng is None:
#         rng = np.random.default_rng()

#     Katt, Krep, rho_0 = create_perturbed_parameters(Katt_std, Krep_std, rho_std, rng)

#     fcbf = FCBF(x0, obstacles, goal, Katt=Katt, Krep=Krep, rho_0=rho_0, d_obs=d_obs, delta=delta)
#     result = solve_ivp(fcbf.fn, (t0, tf_max), x0, max_step=1e-2, dense_output=True, events=fcbf.event)

#     tf = result.t[-1]
#     t_eval = np.linspace(t0, tf, n+1)
#     cpts = result.sol(t_eval)

#     traj = Bernstein(cpts, t0=t0, tf=tf)

#     if debug:
#         return traj, Katt, Krep, rho_0, result
#     else:
#         return traj



def generate_cbf_trajectory(x0, goal, obstacles, obstacle_safe_distances,
                            n=5, Katt=1, Krep=1, rho_0=1, tf_max=60, t0=0, delta=0.01, debug=False):

    fcbf = FCBF(x0, obstacles, obstacle_safe_distances, goal, Katt=Katt, Krep=Krep, rho_0=rho_0, delta=delta)
    result = solve_ivp(fcbf.fn, (t0, tf_max), x0, max_step=1e-2, dense_output=True, events=fcbf.event)

    tf = result.t[-1]
    t_eval = np.linspace(t0, tf, n+1)
    cpts = result.sol(t_eval)

    traj = Bernstein(cpts, t0=t0, tf=tf)

    if debug:
        return traj, Katt, Krep, rho_0, result
    else:
        return traj


def create_perturbed_parameters(Katt_std, Krep_std, rho_std, rng):
    Katt = rng.lognormal(mean=1, sigma=Katt_std)
    Krep = rng.lognormal(mean=1, sigma=Krep_std)
    rho_0 = rng.lognormal(mean=0.5, sigma=rho_std)

    return Katt, Krep, rho_0


class FCBF:
    def __init__(self, x0, obstacles, obstacle_safe_distances, goal, Katt=1, Krep=1, rho_0=1, delta=0.01):
        self.x0 = x0
        self.obstacles = tuple(obstacles)
        self.safe_dists = obstacle_safe_distances
        self.goal = goal
        self.Katt = Katt
        self.Krep = Krep
        self.rho_0 = rho_0
        self.delta = delta

    def make_lsoda_fn_address(self):
        func = make_lsoda_fn(self.goal, self.obstacles, self.safe_dists, self.Katt, self.Krep, self.rho_0, self.delta)

        return func.address

    def fn(self, t, x):
        return _fcbf_fn(t, x, self.goal, self.obstacles, self.safe_dists, self.Katt, self.Krep, self.rho_0, self.delta)

    def event(self, t, x):
        distance = np.linalg.norm(x - self.goal)

        if distance < 1e-3:
            distance = 0.0

        return distance

    event.terminal = True  # Required for the integration to stop early


def make_lsoda_fn(goal, obstacles, safe_dists, k_att, k_rep, rho_0, delta):
    @cfunc(lsoda_sig)
    def _fapf_fn_lsoda(t, x_, dx, _):
        x = carray(x_, (2,))

        vstar = _fcbf_fn(t, x, goal, obstacles, safe_dists, k_att, k_rep, rho_0, delta)

        for i in range(len(vstar)):
            dx[i] = vstar[i]

    return _fapf_fn_lsoda


@njit(cache=True)
def _fcbf_fn(t, x, goal, obstacles, safe_dists, k_att, k_rep, rho_0, delta):
    norm = np.linalg.norm(x - goal)
    vstar = -_grad_uatt(x, k_att, goal) #/ norm

    # min_dist = np.inf
    min_dist = rho_0
    nearest_obstacle = None
    for i, obs in enumerate(obstacles):
        dist = np.linalg.norm(x - obs) - safe_dists[i]
        if dist < min_dist:
            min_dist = dist
            nearest_obstacle = obs
            d_obs = safe_dists[i]

    if nearest_obstacle is not None:
        psi = _psi(x, k_att, k_rep, rho_0, goal, nearest_obstacle, d_obs, delta)
        if psi < 0:
            grad_h = _grad_h(x, k_rep, rho_0, nearest_obstacle, d_obs)
            vstar += -grad_h * psi/(grad_h@grad_h)

    return vstar / norm

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
    return 0.5*k_att*((x-x_goal)@(x-x_goal))


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


# @njit(cache=True)
def solve_ivp_euler(fn, x0, t0, tf, n_steps, terminal_fn=None):
    dt = (tf - t0) / n_steps
    x = [x0]
    for t in np.linspace(t0, tf, n_steps):
        x.append(fn(t, x[-1])*dt + x[-1])
        if terminal_fn is not None and terminal_fn(x[-1]):
            break

    return np.array(x), t


@njit(cache=True)
def fast_generate_cbf_trajectory(x0, goal, obstacles, obstacle_safe_distances,
                                 n_steps=1000, tf_max=60, Katt=1, Krep=1, rho_0=1, delta=0.0, debug=False):

    t0 = 0.0

    # TODO - adaptive steps
    # IVP Solver - Euler's Method
    dt = (tf_max - t0) / n_steps
    t_eval = np.linspace(t0, tf_max, n_steps)
    x = np.zeros((len(t_eval)+1, len(x0)))
    x[0, :] = x0
    for i, t in enumerate(t_eval):
        vstar = _fcbf_fn(t, x[i, :], goal, obstacles, obstacle_safe_distances, Katt, Krep, rho_0, delta)
        x[i+1, :] = vstar*dt + x[i, :]
        if np.linalg.norm(x[-1] - goal) < 1e-2:
            break

    return x, t


if __name__ == '__main__':
    import time

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    ###
    # Problem setup
    ###
    rng_seed = 2
    goal_std = 0.1
    obs_pos_std = 1
    obs_size_std = 0.3
    tf = 50
    goal = np.array([20, 20], dtype=float)
    x0 = np.array([0.1, 0.1], dtype=float)
    obstacles = [np.array([8.5, 5], dtype=float),  # Obstacle positions (m)
                 np.array([1, 4], dtype=float),
                 np.array([7, 1], dtype=float),
                 np.array([8, 12], dtype=float),
                 np.array([50, -3], dtype=float),
                 np.array([80, -3], dtype=float),
                 np.array([30, -1], dtype=float)
                 ]
    obstacle_safe_distances = np.array([1, #6, #1
                                        2,
                                        3, #5, #3
                                        5,
                                        1,
                                        3,
                                        1], dtype=float)

    ###
    # Testing generate_cbf_trajectory
    ###
    rng = np.random.default_rng(rng_seed)
    tstart = time.time()
    for i in range(3):
        obs_tmp = tuple([obs + rng.normal(scale=0.05, size=2) for obs in obstacles])
        obs_dsafe_tmp = tuple([obs_dist + rng.normal(scale=0.01) for obs_dist in obstacle_safe_distances])
        traj = generate_cbf_trajectory(x0, goal, obs_tmp, obs_dsafe_tmp, tf_max=tf, n=30, Katt=1)
    print(f'Computation time for 3 runs (generate_cbf_trajectory): {time.time() - tstart} s')

    ###
    # Euler's IVP method
    ###
    rng = np.random.default_rng(rng_seed)
    results = []
    tstart = time.time()
    for i in range(100):
        obs_tmp = tuple([obs + rng.normal(scale=obs_pos_std, size=2) for obs in obstacles])
        obs_dsafe_tmp = tuple([np.abs(obs_dist + rng.normal(scale=obs_size_std)) for obs_dist in obstacle_safe_distances])
        goal_tmp = goal + rng.normal(scale=goal_std, size=goal.size)
        result, tf_early_term = fast_generate_cbf_trajectory(x0, goal_tmp, obs_tmp, obs_dsafe_tmp,
                                                             n_steps=1000, rho_0=100, Krep=0.1, tf_max=60)
        results.append(result)
    print(f'Computation time for 100 runs (euler ivp): {time.time()-tstart} s')

    ###
    # Numba LSODA's ivp solver
    ###
    fcbf = FCBF(x0, obstacles, obstacle_safe_distances, goal, rho_0=1, delta=0.0)
    t_eval = np.linspace(0, tf, tf*100000)
    fn_address = fcbf.make_lsoda_fn_address()
    tstart = time.time()
    usol, success = lsoda(fn_address, x0, t_eval)
    print(f'Computation time (lsoda): {time.time() - tstart} s')

    ###
    # Plot the results
    ###
    plt.close('all')
    fig0, ax0 = plt.subplots()
    traj.plot(ax0)
    for i, obs in enumerate(obstacles):
        ax0.add_artist(Circle(obs, radius=obstacle_safe_distances[i]))

    fig1, ax1 = plt.subplots()
    for result in results:
        ax1.plot(result[:, 0], result[:, 1], alpha=0.75)
    for i, obs in enumerate(obstacles):
        ax1.add_artist(Circle(obs, radius=obstacle_safe_distances[i]))

    fig2, ax2 = plt.subplots()
    ax2.plot(usol[:, 0], usol[:, 1])
    for i, obs in enumerate(obstacles):
        ax2.add_artist(Circle(obs, radius=fcbf.safe_dists[i]))
