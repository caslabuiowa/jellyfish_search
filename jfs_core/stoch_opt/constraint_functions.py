#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:22:54 2022

@author: magicbycalvin
"""

from abc import ABC, abstractmethod

import numpy as np

from polynomial.bernstein import Bernstein


class ConstraintBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def call(self, trajs):
        pass


class CollisionAvoidance(ConstraintBase):
    # TODO: Modify this class to take obstacle objects rather than a list of points and create an obstacle class that
    # can save relevant information such as a predicted trajectory if it is dynamic and also use the metrics from the
    # paper Pseudospectral motion planning techniques for autonomous obstacle avoidance.
    def __init__(self, safe_dist, obstacles, elev=30):
        self.safe_dist = safe_dist
        self.obstacles = obstacles
        self.elev = elev

    def call(self, trajs):
        for traj in trajs:
            ndim = traj.dim
            n = traj.deg
            t0 = traj.t0
            tf = traj.tf

            for i, obs in enumerate(self.obstacles):
                cpts = np.array([[i]*(n+1) for i in obs[:ndim]], dtype=float)
                c_obs = Bernstein(cpts, t0, tf)

                dist = (traj - c_obs).normSquare().elev(self.elev).cpts

                # if type(self.safe_dist) is list:
                #     if np.any(dist - self.safe_dist[i]**2 < 0):
                #         return False
                # else:
                #     if np.any(dist - self.safe_dist**2 < 0):
                #         return False

                try:
                    result = np.all(dist - self.safe_dist[i]**2 >= 0)
                except TypeError:
                    result = np.all(dist - self.safe_dist**2 >= 0)

        return result


class MaximumSpeed(ConstraintBase):
    def __init__(self, max_speed, elev=10):
        self.max_speed = max_speed
        self.elev = elev

    def call(self, trajs):
        for traj in trajs:
            speed = traj.diff().normSquare().elev(self.elev).cpts

            if np.any(self.max_speed**2 - speed < 0):
                # print('[!] Maximum speed constraint infeasible.')
                return False

        return True


class MaximumAngularRate(ConstraintBase):
    def __init__(self, max_angular_rate, elev=30):
        self.max_angular_rate = max_angular_rate
        self.elev = elev

    def call(self, trajs):
        for traj in trajs:
            cdot = traj.diff()
            cddot = cdot.diff()

            num = cdot.x * cddot.y - cddot.x * cdot.y
            den = cdot.normSquare()
            ang_rate_cpts = (num.elev(self.elev) / (den.elev(self.elev))).cpts

            if np.any(self.max_angular_rate - ang_rate_cpts < 0):
                # print('[!] Maximum angular rate constraint infeasible.')
                return False
            elif np.any(self.max_angular_rate + ang_rate_cpts < 0):
                # print('[!] Minimum angular rate constraint infeasible.')
                return False

        return True


# TODO: Fix this (and possibly others) to only take a single trajectory and extend it to multiple trajectories with a
# new function
class SafeSphere(ConstraintBase):
    def __init__(self, x0, rsafe):
        self.x0 = x0.squeeze()[..., np.newaxis]
        self.rsafe = rsafe

    def call(self, trajs):
        for i, traj in enumerate(trajs):
            # x0 = self.x0[i].reshape(-1, 1)
            if np.any(np.linalg.norm(traj.cpts - self.x0, axis=0) > self.rsafe):
                # print('[!] Safe sphere constraint infeasible.')
                return False

        return True


class MinimumFinalTime(ConstraintBase):
    def __init__(self, tf_min):
        self.tf_min = tf_min

    def call(self, trajs):
        if np.any([i.tf < self.tf_min for i in trajs]):
            # print('[!] Minimum final time constraint infeasible.')
            return False
        else:
            return True


if __name__ == '__main__':
    pass
