#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:23:03 2022

@author: magicbycalvin
"""

from abc import ABC, abstractmethod
import types

import numpy as np

from BeBOT.polynomial.bernstein import Bernstein


class CostBase(ABC):
    def __init__(self, weight=1.0):
        self._weight = weight

    def __add__(self, other_cost):
        return CombinedCost([self, other_cost])

    def __mul__(self, val):
        self._weight = val
        return self

    __rmul__ = __mul__

    @abstractmethod
    def _call(self, trajs):
        pass

    def call(self, trajs):
        """Do not override this method, instead override _call

        Parameters
        ----------
        trajs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self._weight*self._call(trajs)


class CombinedCost(CostBase):
    def __init__(self, costs):
        super().__init__()
        self.costs = costs

    def _call(self, trajs):
        return sum([cost.call(trajs) for cost in self.costs])


class MaximumJerk(CostBase):
    def __init__(self):
        super().__init__()

    def _call(self, trajs):
        jerks = []
        for traj in trajs:
            max_jerk = traj.diff().diff().diff().normSquare().cpts.max()
            jerks.append(max_jerk)

        return max(jerks)


class SumOfJerk(CostBase):
    def __init__(self):
        super().__init__()

    def _call(self, trajs):
        jerks = []
        for traj in trajs:
            jerk = traj.diff().diff().diff().normSquare().integrate()
            jerks.append(jerk)

        return sum(jerks)


class MaximumAcceleration(CostBase):
    def __init__(self):
        super().__init__()

    def _call(self, trajs):
        accelerations = []
        for traj in trajs:
            max_acceleration = traj.diff().diff().diff().normSquare().cpts.max()
            accelerations.append(max_acceleration)

        return max(accelerations)


class SumOfAcceleration(CostBase):
    def __init__(self):
        super().__init__()

    def _call(self, trajs):
        accelerations = []
        for traj in trajs:
            acceleration = traj.diff().diff().diff().normSquare().integrate()
            accelerations.append(acceleration)

        return sum(accelerations)


class DistanceToGoal(CostBase):
    def __init__(self, goal):
        super().__init__()
        self.goal = goal.squeeze()

    def _call(self, trajs):
        cost = [np.linalg.norm(self.goal - i.cpts[:, -1].squeeze()) for i in trajs]

        return sum(cost)


class SumOfDistance(CostBase):
    def __init__(self, goal):
        super().__init__()
        self.goal = goal.squeeze()

    def _call(self, trajs):
        distances = []
        for traj in trajs:
            ndim = traj.dim
            n = traj.deg
            t0 = traj.t0
            tf = traj.tf

            cpts = np.array([[i]*(n+1) for i in self.goal[:ndim]], dtype=float)
            c_goal = Bernstein(cpts, t0, tf)

            dist = (traj - c_goal).normSquare().integrate()
            distances.append(dist)

        return sum(distances).squeeze()


if __name__ == '__main__':
    pass
