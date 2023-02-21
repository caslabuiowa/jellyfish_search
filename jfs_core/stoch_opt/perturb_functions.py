#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 20:01:11 2022

@author: magicbycalvin
"""

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from numpy.random import default_rng

from polynomial.bernstein import Bernstein
from .utils import state2cpts


class PerturbBase(ABC):
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


class GaussianPerturb(PerturbBase):
    def __init__(self, rng_seed=None, rng=None, std=1.0):
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
