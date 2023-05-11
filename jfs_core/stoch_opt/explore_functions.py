#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:39:04 2022

@author: magicbycalvin
"""

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from numpy.random import default_rng

from BeBOT.polynomial.bernstein import Bernstein
from .utils import state2cpts


class ExploreBase(ABC):
    def __init__(self, rng_seed=None, rng=None):
        if rng is not None:
            if rng_seed is not None:
                print('[!] Warning: The RNG seed provided will be unused since an RNG was provided.')
            self.rng = rng

        else:
            self.rng = default_rng(rng_seed)

    @abstractmethod
    def call(self, trajs):
        pass


class GaussianExploration(ExploreBase):
    def __init__(self, rng_seed=None, rng=None, std=1.0):
        """Create a deep copy of the trajectory and perturb the control points and final time with Gaussian noise.

        This function assumes that the initial position, velocity, and acceleration are all fixed. Therefore, the first
        three control points in each dimension are not perturbed.

        Parameters
        ----------
        trajectory : Bernstein
            Trajectory whose control points and final time will be perturbed.
        std : float
            Standard deviation of the Gaussian noise used to perturb the control points and final time.
        rng : np.random.default_rng, optional
            Random number generator object for generating the Gaussian noise. If no generator is passed in, the
            function will create its own. The default is None.

        Returns
        -------
        trajectory : Bernstein
            The resulting perturbed trajectory.
        """
        super().__init__(rng_seed=rng_seed, rng=rng)
        self.std = std

    def call(self, trajs):
        new_trajs = []
        for traj in trajs:
            new_traj = deepcopy(traj)
            ndim = traj.dim
            n = traj.deg
            x0 = traj.cpts[:, 0]
            x0dot = traj.diff().cpts[:, 0]
            x0ddot = traj.diff().diff().cpts[:, 0]

            new_traj.tf = self.rng.normal(new_traj.tf, self.std)
            if new_traj.tf < new_traj.t0:
                new_traj.tf = new_traj.t0 + 1e-3

            t0 = new_traj.t0
            tf = new_traj.tf

            new_traj.cpts[:, :3] = state2cpts(x0, x0dot, x0ddot, t0, tf, n)
            new_traj.cpts[:, 3:] += self.rng.normal(scale=self.std, size=(ndim, n-2))

            new_trajs.append(new_traj)

        return new_trajs


if __name__ == '__main__':
    pass
