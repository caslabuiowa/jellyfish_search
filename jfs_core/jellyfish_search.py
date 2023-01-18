#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 12:01:18 2022

@author: magicbycalvin
"""

import bisect
import logging

from control import lqr
import matplotlib.pyplot as plt
from numba import cfunc, carray
from numbalsoda import lsoda_sig, lsoda
import numpy as np
from numpy.random import default_rng

from polynomial.bernstein import Bernstein
from stoch_opt.constraint_functions import CollisionAvoidance, MaximumSpeed, MaximumAngularRate, SafeSphere
from stoch_opt.cost_functions import SumOfDistance
from stoch_opt.utils import state2cpts

from lqr_search import generate_lqr_trajectory

# TODO: Major refactoring required since the functions and objects are incredibly messy

LOG_LEVEL = logging.WARN
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
if len(logger.handlers) < 1:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LOG_LEVEL)
    logger.addHandler(stream_handler)


class Worker:
    def __init__(self, trajectory_gen_fn, n_trajectories, obstacles, rsafe):
        self.running = False
        self.trajectories = []

        self.generate_trajectory = trajectory_gen_fn
        self.n_trajectories = n_trajectories
        self.obstacles = obstacles
        self.rsafe = rsafe
        self.solver_params = {}

    def do_work(self, safe_dist, vmax, wmax):
        self.running = True
        for i in range(self.n_trajectories):
            temp_traj = self.generate_trajectory()
            logger.debug('Checking feasibility')
            if is_feasible(temp_traj, self.obstacles, safe_dist, vmax, wmax, self.rsafe):
                logger.debug('Assigning cost')
                cost = cost_fn(temp_traj, self.solver_params['goal'])
                bisect.insort(self.trajectories, (temp_traj, cost), key=lambda x: x[1])

        self.running = False


# TODO: determine whether to use t_max or n_trajectories, or figure out how to offer both
# TODO: add a function to add/remove obstacles
class JellyfishController:
    def __init__(self, n, ndim, tf, obstacles, safe_dist, vmax, wmax, safe_planning_radius, t_max, n_trajectories,
                 solver_params={}, rng_seed=None):
        self.n = n
        self.ndim = ndim
        self.tf = tf
        self.obstacles = obstacles
        self.safe_dist = safe_dist
        self.vmax = vmax
        self.wmax = wmax
        self.safe_planning_radius = safe_planning_radius
        self.t_max = t_max
        self.n_trajectories = n_trajectories
        self.solver_params = solver_params
        self.rng_seed = rng_seed
        self.rng = default_rng(rng_seed)

    def run_worker(self, solver_params):
        solver_method = solver_params['method']
        def trajectory_gen_fn():
            if solver_method.lower() == 'lqr':
                return generate_lqr_trajectory(solver_params['x0'],
                                               solver_params['goal'],
                                               solver_params['Q_std'],
                                               solver_params['R_std'],
                                               solver_params['x0_std'],
                                               solver_params['tf'],
                                               n=self.n,
                                               rng=self.rng)
            elif solver_method.lower() == 'brachistochrone':
                pass
            elif solver_method.lower() == 'least_action':
                pass
            else:
                raise ValueError('Incorrect solver method.')

        self.worker = Worker(trajectory_gen_fn, self.n_trajectories, self.obstacles, self.safe_planning_radius)
        self.worker.solver_params = solver_params
        self.worker.do_work(self.safe_dist, self.vmax, self.wmax)

        return self.worker.trajectories


# TODO: create and integrate the solver and problem parameters classes instead of using a dictionary
# TODO: streamline the locations where parameters are input to the problem
class SolverParameters:
    def __init__(self, x0, goal, tf, method):
        self.x0 = x0
        self.goal = goal
        self.tf = tf
        self.method = method


class ProblemParameters:
    def __init__(self):
        pass


def is_feasible(traj, obstacles, safe_dist, vmax, wmax, rsafe):
    logger.debug('Inside is_feasible')
    constraints = [
        CollisionAvoidance(safe_dist, obstacles, elev=100),
        MaximumSpeed(vmax),
        MaximumAngularRate(wmax),
        SafeSphere(traj.cpts[:, 0].squeeze(), rsafe)
    ]

    logger.debug('Starting feasibility check')
    for i, cons in enumerate(constraints):
        logger.debug(f'Constraint {i}')
        if not cons.call([traj]):
            return False

    return True


def cost_fn(traj, goal):
    sod = SumOfDistance(goal)

    return sod.call([traj])


def project_goal(x, rsafe, goal):
    x = x.squeeze()
    goal = goal.squeeze()
    u = np.linalg.norm(goal - x)
    scale = rsafe / u

    if scale < 1:
        new_goal = (goal - x)*scale + x
    else:
        new_goal = goal

    return new_goal


if __name__ == '__main__':
    seed = 3
    Q_std = 1
    R_std = 1
    x0_std = 1

    n = 5
    t0 = 0
    tf = 10
    # dt = 1
    ndim = 2
    t_max = 0.95
    n_trajectories = 1000

    safe_planning_radius = 10
    safe_dist = 1
    vmax = 3
    wmax = np.pi/4

    obstacles = [np.array([15, 0], dtype=float),  # Obstacle positions (m)
                 np.array([20, 2], dtype=float),
                 np.array([60, 1], dtype=float),
                 np.array([40, 2], dtype=float),
                 np.array([50, -3], dtype=float),
                 np.array([80, -3], dtype=float),
                 np.array([30, -1], dtype=float)]

    goal = np.array([100, 0], dtype=float)
    initial_position = np.array([0, 0], dtype=float)
    initial_velocity = np.array([1, 0], dtype=float)
    initial_acceleration = np.array([0.1, 1], dtype=float)

    cur_goal = project_goal(initial_position, safe_planning_radius, goal)
    x0 = np.array([initial_position[0],
                   initial_velocity[0],
                   initial_acceleration[0],
                   0,
                   initial_position[1],
                   initial_velocity[1],
                   initial_acceleration[1],
                   0], dtype=float)

    solver_params = {'method': 'lqr',
                     'x0': x0,
                     'goal': cur_goal,
                     'tf': tf,
                     'Q_std': Q_std,
                     'R_std': R_std,
                     'x0_std': x0_std}

    jfc = JellyfishController(n, ndim, tf, obstacles, safe_dist, vmax, wmax, safe_planning_radius, t_max,
                              n_trajectories, rng_seed=seed)
    trajs = jfc.run_worker(solver_params)

    plt.close('all')
    fig, ax = plt.subplots()
    for traj in trajs:
        traj[0].plot(ax, showCpts=False)
