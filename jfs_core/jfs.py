#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:39:45 2023

@author: magicbycalvin
"""

import multiprocessing as mp

import numpy as np

from BeBOT.polynomial.bernstein import Bernstein

from jfs_core.cbf_search import fast_generate_cbf_trajectory, fast_generate_cbf_trajectory_no_obs
from jfs_core.stoch_opt.constraint_functions import CollisionAvoidance, MaximumAngularRate, MaximumSpeed, SafeSphere
from jfs_core.stoch_opt.cost_functions import SumOfDistance


class JellyfishSearch:
    def __init__(self, rng_seed=None, num_workers=None, discrete_traj=False):
        self._pool = mp.Pool(num_workers)
        self._rng_seed = rng_seed
        self._discrete_traj = discrete_traj

    def __del__(self):
        self._pool.terminate()

    def generate_trajectories(self, x0, goal, obstacles, obstacle_safe_distances, num_trajectories, vmax, wmax, rsafe,
                              obs_pos_std, obs_size_std, goal_pos_std, degree,
                              solver_params={}):
        seed_seq = np.random.SeedSequence(self._rng_seed)
        rng_seeds = seed_seq.spawn(num_trajectories)
        print(f'Current goal from JFS: {goal}')
        print(f'Goal position std: {goal_pos_std}')
        results = self._pool.starmap(_traj_gen_wrapper, zip([x0]*num_trajectories,
                                                            [goal]*num_trajectories,
                                                            [obstacles]*num_trajectories,
                                                            [obstacle_safe_distances]*num_trajectories,
                                                            rng_seeds,
                                                            [obs_pos_std]*num_trajectories,
                                                            [obs_size_std]*num_trajectories,
                                                            [goal_pos_std]*num_trajectories,
                                                            [vmax]*num_trajectories,
                                                            [wmax]*num_trajectories,
                                                            [rsafe]*num_trajectories,
                                                            [degree]*num_trajectories,
                                                            [self._discrete_traj]*num_trajectories,
                                                            [solver_params]*num_trajectories))

        results.sort(key=lambda x: x[1])

        return results

    # def generate_trajectories_pool(self, x0, goal, obstacles, obstacle_safe_distances, num_trajectories,
    #                                obs_pos_std, obs_size_std, goal_pos_std,
    #                                solver_params={}):
    #     seed_seq = np.random.SeedSequence(self._rng_seed)
    #     rng_seeds = seed_seq.spawn(num_trajectories)

    #     with mp.Pool(10) as pool:
    #         results = pool.starmap(_traj_gen_wrapper, zip([x0]*num_trajectories,
    #                                                       [goal]*num_trajectories,
    #                                                       [obstacles]*num_trajectories,
    #                                                       [obstacle_safe_distances]*num_trajectories,
    #                                                       rng_seeds,
    #                                                       [obs_pos_std]*num_trajectories,
    #                                                       [obs_size_std]*num_trajectories,
    #                                                       [goal_pos_std]*num_trajectories,
    #                                                       [solver_params]*num_trajectories))

    #     return results


def _traj_gen_wrapper(x0, goal, obstacles, obstacle_safe_distances, rng_seed, obs_pos_std, obs_size_std, goal_pos_std,
                      vmax, wmax, rsafe, degree, discrete, solver_params):
    try:
        rng = np.random.default_rng(rng_seed)
        goal_tmp = goal + rng.normal(scale=goal_pos_std, size=goal.size)
        if len(obstacles[0]) > 0:
            obs_tmp = np.array([obs + rng.normal(scale=obs_pos_std, size=2) for obs in obstacles])
            obs_dsafe_tmp = np.array([obs_dist + np.abs(rng.normal(scale=obs_size_std))
                                   for obs_dist in obstacle_safe_distances])
            discrete_traj = fast_generate_cbf_trajectory(x0, goal_tmp, obs_tmp, obs_dsafe_tmp, **solver_params)
        else:
            discrete_traj = fast_generate_cbf_trajectory_no_obs(x0, goal_tmp, **solver_params)

        if discrete:
            return discrete_traj

        traj = _discrete_to_bernstein(discrete_traj, degree)

        if _feasibility_check(traj, obstacle_safe_distances, obstacles, vmax, wmax, rsafe):
            cost = _cost_fn(traj, goal)
        else:
            cost = np.inf
    except ZeroDivisionError as e:
        print(f'[!] Warning: {e}, generating new trajectory.')
        traj, cost = _traj_gen_wrapper(x0, goal, obstacles, obstacle_safe_distances, rng_seed.spawn(1)[0], obs_pos_std,
                                       obs_size_std, goal_pos_std, vmax, wmax, rsafe, degree, discrete, solver_params)

    return (traj, cost)


def _feasibility_check(traj, safe_distances, obstacles, vmax, wmax, rsafe):
    constraints = [
        CollisionAvoidance(safe_distances, obstacles, elev=20),
        MaximumSpeed(vmax),
        MaximumAngularRate(wmax),
        SafeSphere(traj.cpts[:, 0].squeeze(), rsafe)
    ]
    feasible = True

    for cons in constraints:
        if not cons.call([traj]):
            feasible = False

    return feasible


def _cost_fn(traj, goal):
    sod = SumOfDistance(goal)

    return sod.call([traj])


def _discrete_to_bernstein(traj, degree):
    idxs = np.linspace(0, len(traj[0])-1, degree+1)
    idxs = [int(i) for i in idxs.round()]

    cpts = traj[0][idxs, :].T
    tf = traj[1]

    if tf < 1e-1:
        tf = 1e-1

    traj_bp = Bernstein(cpts=cpts, t0=0.0, tf=tf)

    return traj_bp


if __name__ == '__main__':
    pass
