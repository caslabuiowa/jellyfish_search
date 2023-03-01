#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 12:01:18 2022

@author: magicbycalvin
"""

import bisect
from dataclasses import dataclass
import logging
from multiprocessing import Pool

from control import lqr
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from numba import cfunc, carray
from numbalsoda import lsoda_sig, lsoda
import numpy as np
from numpy.random import SeedSequence, default_rng
from numpy.typing import ArrayLike

from polynomial.bernstein import Bernstein
from stoch_opt.constraint_functions import CollisionAvoidance, MaximumSpeed, MaximumAngularRate, SafeSphere
from stoch_opt.cost_functions import SumOfDistance
from stoch_opt.utils import state2cpts

from apf_search import generate_apf_trajectory, generate_piecewise_apf_trajectory
from brachistochrone_search import generate_brachistochrone_trajectory
from lqr_search import generate_lqr_trajectory

# TODO: Major refactoring required since the functions and objects are incredibly messy

LOG_LEVEL = logging.WARNING
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
if len(logger.handlers) < 1:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LOG_LEVEL)
    logger.addHandler(stream_handler)


DISABLE_CONSTRAINTS = False


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
            print(f'Trajectory {i}')
            temp_traj = self.generate_trajectory()
            logger.debug('Checking feasibility')
            if is_feasible(temp_traj, self.obstacles, safe_dist, vmax, wmax, self.rsafe) or DISABLE_CONSTRAINTS:
                logger.debug('Assigning cost')
                cost = cost_fn(temp_traj, self.solver_params['goal'])
                bisect.insort(self.trajectories, (temp_traj, cost), key=lambda x: x[1])

        self.running = False


class PWWorker:
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
            print(f'Trajectory {i}')
            temp_pw_traj = self.generate_trajectory()
            logger.debug('Checking feasibility')

            traj_feasibility = []
            for traj in temp_pw_traj:
                traj_feasibility.append(is_feasible(traj, self.obstacles, safe_dist, vmax, wmax, self.rsafe))

            if np.all(traj_feasibility) or DISABLE_CONSTRAINTS:
                logger.debug('Assigning cost')
                cost = sum([cost_fn(traj, self.solver_params['goal']) for traj in temp_pw_traj])
                bisect.insort(self.trajectories, (temp_pw_traj, cost), key=lambda x: x[1])

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
                return generate_brachistochrone_trajectory(solver_params['x0'],
                                                           solver_params['goal'],
                                                           solver_params['R_std'],
                                                           n=self.n,
                                                           rng=self.rng)
            elif solver_method.lower() == 'apf':
                return generate_apf_trajectory(solver_params['x0'],
                                               solver_params['goal'],
                                               solver_params['obstacles'],
                                               rho_std=0.1,
                                               tf_max=600,
                                               n=self.n,
                                               rng=self.rng)
            elif solver_method.lower() == 'apf_pw':
                return generate_piecewise_apf_trajectory(solver_params['x0'],
                                                         solver_params['goal'],
                                                         solver_params['obstacles'],
                                                         rho_std=0.1,
                                                         tf_max=600,
                                                         n=self.n,
                                                         rng=self.rng)
            else:
                raise ValueError('Incorrect solver method.')

        if solver_method.lower() == 'apf_pw':
            self.worker = PWWorker(trajectory_gen_fn, self.n_trajectories, self.obstacles, self.safe_planning_radius)
        else:
            self.worker = Worker(trajectory_gen_fn, self.n_trajectories, self.obstacles, self.safe_planning_radius)
        self.worker.solver_params = solver_params
        self.worker.do_work(self.safe_dist, self.vmax, self.wmax)

        return self.worker.trajectories


# TODO: create and integrate the solver and problem parameters classes instead of using a dictionary
# TODO: streamline the locations where parameters are input to the problem
@dataclass
class SolverParameters:
    method: str
    params: dict
    trajectory_count: int
    rng_seed: int | None = None
    process_count: int | None = None


@dataclass
class ProblemParameters:
    n: int
    ndim: int
    x0: ArrayLike
    goal: ArrayLike
    tf: float
    obstacles: ArrayLike
    safe_distance: float
    maximum_speed: float
    maximum_angular_rate: float
    safe_planning_radius: float


def generate_jellyfish_trajectory(problem_parameters: ProblemParameters, solver_parameters: SolverParameters,
                                  debug=False):



    seed_seq = SeedSequence(solver_parameters.rng_seed)
    rng_seeds = seed_seq.spawn(solver_parameters.trajectory_count)
    with Pool(solver_parameters.process_count) as pool:
        # trajectories = pool.map(function_wrapper, rng_seeds)
        trajectories = pool.starmap(function_wrapper, zip(rng_seeds,
                                                          [problem_parameters]*len(rng_seeds),
                                                          [solver_parameters]*len(rng_seeds),
                                                          [debug]*len(rng_seeds)))

    return trajectories


def function_wrapper(rng_sequence_seed, problem_parameters, solver_parameters, debug):
    traj = _trajectory_gen_fn(rng_sequence_seed, problem_parameters, solver_parameters, debug)
    cost = _assess_trajectory(traj, problem_parameters)
    return (traj, cost)


def _trajectory_gen_fn(rng_sequence_seed, problem_parameters, solver_parameters, debug):
    rng = default_rng(rng_sequence_seed)

    solver_method = solver_parameters.method.lower()
    if solver_method == 'apf':
        return generate_apf_trajectory(problem_parameters.x0,
                                       problem_parameters.goal,
                                       problem_parameters.obstacles,
                                       rho_std=solver_parameters.params['rho_std'],
                                       tf_max=solver_parameters.params['tf_max'],
                                       n=problem_parameters.n,
                                       rng=rng)
    elif solver_method == 'apf_pw':
        pass
    elif solver_method == 'brachistochrone':
        pass
    elif solver_method == 'lqr':
        return generate_lqr_trajectory(problem_parameters.x0,
                                       problem_parameters.goal,
                                       solver_parameters.params['Q_std'],
                                       solver_parameters.params['R_std'],
                                       solver_parameters.params['x0_std'],
                                       problem_parameters.tf,
                                       n=problem_parameters.n,
                                       rng=rng)
    else:
        raise ValueError('Incorrect solver method.')


def _assess_trajectory(traj, problem_parameters:ProblemParameters):
    obstacles = problem_parameters.obstacles
    safe_dist = problem_parameters.safe_distance
    vmax = problem_parameters.maximum_speed
    wmax = problem_parameters.maximum_angular_rate
    rsafe = problem_parameters.safe_planning_radius

    if is_feasible(traj, obstacles, safe_dist, vmax, wmax, rsafe):
        cost = cost_fn(traj, goal)
    else:
        cost = np.inf

    return cost


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
            logger.debug('--> Infeasible')
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
    import time
    seed = 3
    Q_std = 1000
    R_std = 1000
    x0_std = 1

    n = 5
    t0 = 0
    tf = 30
    # dt = 1
    ndim = 2
    t_max = 0.95
    n_trajectories = 1000

    safe_planning_radius = 11
    safe_distance = 1
    maximum_speed = 3
    maximum_angular_rate = np.pi/4

    obstacles = [np.array([5, 0], dtype=float),  # Obstacle positions (m)
                 np.array([20, 2], dtype=float),
                 np.array([60, 1], dtype=float),
                 np.array([40, 2], dtype=float),
                 np.array([50, -3], dtype=float),
                 np.array([80, -3], dtype=float),
                 np.array([30, -1], dtype=float)]

    goal = np.array([100, 0], dtype=float)
    initial_position = np.array([0, 0.1], dtype=float)
    initial_velocity = np.array([1, 0], dtype=float)
    initial_acceleration = np.array([0.1, 1], dtype=float)

    cur_goal = project_goal(initial_position, safe_planning_radius-1, goal)
    x0 = np.array([initial_position[0],
                   # initial_velocity[0],
                   # initial_acceleration[0],
                   # 0,
                   initial_position[1],
                   # initial_velocity[1],
                   # initial_acceleration[1],
                   # 0
                   ], dtype=float)
    # x0 = initial_position

    solver_params = SolverParameters('apf',
                                     {'rho_std': 0.1,
                                      'tf_max': 600},
                                     n_trajectories)
    problem_params = ProblemParameters(n,
                                       ndim,
                                       x0,
                                       cur_goal,
                                       tf,
                                       obstacles,
                                       safe_distance,
                                       maximum_speed,
                                       maximum_angular_rate,
                                       safe_planning_radius)

    tik = time.time()
    traj_list = generate_jellyfish_trajectory(problem_params, solver_params)
    print(f'Elapsed time: {time.time() - tik} s')

    plt.close('all')
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    for traj in traj_list:
        if traj[1] < np.inf:
            traj[0].plot(ax, showCpts=False, color='g', alpha=0.5)
            traj[0].diff().normSquare().plot(ax2, showCpts=False, color='g')
        else:
            traj[0].plot(ax, showCpts=False, color='r', alpha=0.2)
            traj[0].diff().normSquare().plot(ax2, showCpts=False, color='r')
    ax.set_title('Trajectories')
    ax2.set_title('Trajectories\' Speed Squared')

    for obs in obstacles:
        artist = Circle(obs, radius=safe_distance)
        ax.add_artist(artist)

    # solver_params = {'method': 'apf_pw',
    #                  'x0': x0,
    #                  'goal': cur_goal,
    #                  'tf': tf,
    #                  'Q_std': Q_std,
    #                  'R_std': R_std,
    #                  'x0_std': x0_std,
    #                  'obstacles': obstacles}

    # jfc = JellyfishController(n, ndim, tf, obstacles, safe_dist, vmax, wmax, safe_planning_radius, t_max,
    #                           n_trajectories, rng_seed=seed)
    # trajs = jfc.run_worker(solver_params)

    # plt.close('all')
    # fig, ax = plt.subplots()
    # if solver_params['method'] == 'apf_pw':
    #     for pw_traj in trajs:
    #         for traj in pw_traj[0]:
    #             traj.plot(ax, showCpts=False)
    # else:
    #     for traj in trajs:
    #         traj[0].plot(ax, showCpts=False)

    # for obs in obstacles:
    #     artist = Circle(obs, radius=safe_dist)
    #     ax.add_artist(artist)

    # fig2, ax2 = plt.subplots()
    # if solver_params['method'] == 'apf_pw':
    #     for pw_traj in trajs:
    #         for traj in pw_traj[0]:
    #             traj.diff().normSquare().plot(ax2, showCpts=False)
    # else:
    #     for traj in trajs:
    #         traj[0].diff().normSquare().plot(ax2, showCpts=False)
