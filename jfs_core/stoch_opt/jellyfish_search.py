#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 17:27:18 2022

@author: magicbycalvin
"""

from copy import deepcopy
import time

import numpy as np
from numpy.random import default_rng
from scipy.stats.qmc import Sobol

# from stoch_opt.explore_functions import GaussianExploration
# from stoch_opt.perturb_functions import GaussianPerturb
from .explore_functions import GaussianExploration
from .perturb_functions import GaussianPerturb
from .utils import state2cpts


class JellyfishSearch:
    def __init__(self, cost_fn, initial_guess, constraints=[],
                 nchildren=3, exploration_prob=0.0, max_time=1.0, explore_fn=None, rng_seed=None, perturb_fn=None,
                 max_iter=1e4, random_walk_std=1.0, goal_eps=1e-3  # These params not implemented yet
                 ):
        # TODO: Implement early stopping conditions

        # Assuming initial_guess will be a list of Bernstein trajectories
        self.cost_fn = cost_fn
        self.q0 = initial_guess
        self.constraints = constraints
        self.nchildren = nchildren
        self.p_exp = exploration_prob
        self.tmax = max_time

        # These parameters are not implemented yet
        # self.goal_eps = goal_eps
        # self.rw_std = random_walk_std
        # self.max_iter = max_iter

        self.feasible_trajs = []
        self.infeasible_trajs = []

        self.rng = default_rng(rng_seed)

        if explore_fn is None:
            exp = GaussianExploration(rng=self.rng, std=10*random_walk_std)
            self.explore_fn = exp

        if perturb_fn is None:
            per = GaussianPerturb(rng=self.rng, std=random_walk_std)
            self.perturb_fn = per

    def solve(self, time_fn=None, print_iter=False):
        if time_fn is None:
            def time_fn(): return time.time()

        if print_iter:
            iter_count = 0

        qstar = self.q0
        if self.feasibility_check(self.q0):
            qstar_cost = self.cost_fn.call(self.q0)
            self.feasible_trajs.append((qstar, qstar_cost))

        else:
            qstar_cost = np.inf
            self.infeasible_trajs.append((qstar, qstar_cost))

        tstart = time_fn()
        while time_fn() - tstart < self.tmax:
            if print_iter:
                print(f'Iteration: {iter_count}\n---> Cost: {qstar_cost}')
                iter_count += 1

            r = self.rng.uniform()
            if r < self.p_exp:
                q = self.explore_fn.call(qstar)

                if self.feasibility_check(q):
                    qcost = self.cost_fn.call(q)
                    self.feasible_trajs.append((q, qcost))

                    if qcost < qstar_cost:
                        qstar = q
                        qstar_cost = qcost
                else:
                    self.infeasible_trajs.append((q, np.inf))

            for i in range(self.nchildren):
                qsample = self.perturb_fn.call(qstar)
                if self.feasibility_check(qsample):
                    qcost = self.cost_fn.call(qsample)
                    self.feasible_trajs.append((qsample, qcost))

                    if qcost < qstar_cost:
                        qstar = qsample
                        qstar_cost = qcost
                else:
                    self.infeasible_trajs.append((qsample, np.inf))

        if print_iter:
            print('---')
            print(f'Computation Time: {time_fn() - tstart}')
            print(f'Final cost: {qstar_cost}')
            print('---')

        return qstar, qstar_cost

    def feasibility_check(self, trajs):
        feasible = True

        for cons in self.constraints:
            if not cons.call(trajs):
                feasible = False

        return feasible


def _perturb_trajectory(trajectory, std, rng=None):
    """Create a deep copy of the given trajectory and perturb the control points and final time with Gaussian noise.

    This function assumes that the initial position, velocity, and acceleration are all fixed. Therefore, the first
    three control points in each dimension are not perturbed.

    Parameters
    ----------
    trajectory : Bernstein
        Trajectory whose control points and final time will be perturbed.
    std : float
        Standard deviation of the Gaussian noise used to perturb the control points and final time.
    rng : np.random.default_rng, optional
        Random number generator object for generating the Gaussian noise. If no generator is passed in, the function
        will create its own. The default is None.

    Returns
    -------
    trajectory : Bernstein
        The resulting perturbed trajectory.
    """
    if rng is None:
        rng = default_rng()
    trajectory = deepcopy(trajectory)
    ndim = trajectory.dim
    n = trajectory.deg

    trajectory.cpts[:, 3:] += rng.normal(scale=std, size=(ndim, n-2))

    trajectory.tf = rng.normal(trajectory.tf, std)
    if trajectory.tf < trajectory.t0:
        trajectory.tf = trajectory.t0 + 1e-3

    return trajectory


def _qmc(x0, t0, tf, goal, n=7, ndim=2, nchildren=8, std=1.0, rng_seed=None):
    # TODO: apply rotation matrix so that the x direction faces directly towards the goal
    x = x0[:ndim]
    xdot = x0[ndim:2*ndim]
    xddot = x0[2*ndim:3*ndim]
    m = int(np.log(nchildren)/np.log(2)) + int(np.log(n-3)/np.log(2))

    rng = default_rng(seed=rng_seed)
    x2goal = goal - x
    x2goal_mag = np.linalg.norm(x2goal)

    cpts = np.empty((ndim, n+1))
    cpts[:, :3] = state2cpts(x, xdot, xddot, t0, tf, n)

    # Sobol exploration for the intermediate points
    qmc_sobol = Sobol(ndim, seed=rng_seed)
    qmc_values = qmc_sobol.random_base2(m)

    # Reshape the Sobol exploration
    # TODO: ROTATION MATRIX! Keep X pointed at the goal, all our X values will need to be increasing while all our
    # Y (and Z for 3D) should stay near each other
    values = []
    for i in range(ndim):
        if i == 0:
            sample_pts = np.sort(qmc_values[:, i]).reshape((-1, nchildren))*x2goal_mag + x[i]
        else:
            sample_pts = np.sort(qmc_values[:, i]).reshape((nchildren, -1))*x2goal_mag - x2goal_mag/2 + x[i]

        for row in sample_pts:
            rng.shuffle(row)

        if i != 0:
            sample_pts = sample_pts.T

        values.append(sample_pts)

    # Gaussian exploration for the final point
    cpts_list = []
    for i in range(nchildren):
        for j, pts in enumerate(values):
            cpts[j, 3:-1] = pts[:, i]
            cpts[j, -1] = rng.normal(loc=goal[j], scale=std)
        cpts_list.append(cpts.copy())

    return cpts_list


if __name__ == '__main__':
    # TODO: In BeBOT, make it possible to create a polynomial from the initial state (i.e., create a polynomial given
    # initial position, velocity, and acceleration). This could be added as a BeBOT utility function
    pass
