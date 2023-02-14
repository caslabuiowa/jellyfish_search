#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:05:18 2023

@author: magicbycalvin
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:31:49 2022

@author: magicbycalvin
"""

import logging

from control import lqr
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from numba import cfunc, carray
from numbalsoda import lsoda_sig, lsoda
import numpy as np
from numpy.random import default_rng
from scipy.integrate import solve_ivp

from polynomial.bernstein import Bernstein

LOG_LEVEL = logging.WARN
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
if len(logger.handlers) < 1:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LOG_LEVEL)
    logger.addHandler(stream_handler)


def generate_apf_trajectory(x0, goal, obstacles,
                            n=5, Katt_std=1, Krep_std=1, rho_std=1, d_obs=1, tf_max=60, t0=0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    Katt, Krep, rho_0 = create_perturbed_parameters(Katt_std, Krep_std, rho_std, rng)

    func_apf = FAPF(x0, obstacles, goal, Katt=Katt, Krep=Krep, rho_0=rho_0, d_obs=d_obs)

    # Note that max step is important otherwise the solver will return a straight line
    res = solve_ivp(func_apf.fn, (t0, tf_max), x0, method='LSODA', max_step=1e-1,
                    events=func_apf.event, dense_output=True)
    tf = res.t[-1]
    t = np.linspace(t0, tf, n)
    cpts = res.sol(t)

    traj = Bernstein(cpts, t0=t0, tf=tf)

    return traj

def create_perturbed_parameters(Katt_std, Krep_std, rho_std, rng):
    Katt = rng.lognormal(mean=1, sigma=Katt_std)
    Krep = rng.lognormal(mean=1, sigma=Krep_std)
    rho_0 = rng.lognormal(mean=0.5, sigma=rho_std)

    return Katt, Krep, rho_0


class FAPF:
    def __init__(self, x0, obstacles, goal, Katt=1, Krep=1, rho_0=1, d_obs=1, t0=0, d=1e-3,
                 disable_rho0=False, gain_fn=None, goal2=None):
        self.x0 = x0
        self.obstacles = obstacles
        self.goal = goal
        self.Katt = Katt
        self.Krep = Krep
        self.rho_0 = rho_0
        self.d_obs = d_obs
        self.t0 = t0
        self.d = d
        self.disable_rho0 = disable_rho0
        self.gain_fn = gain_fn
        self.goal2 = goal2

    def fn(self, t, x):
        norm = np.linalg.norm(x.squeeze() - self.goal) + self.d
        # print(f'{norm=}')
        u_att_x = self.Katt*(x[0] - self.goal[0]) / norm
        u_att_y = self.Katt*(x[1] - self.goal[1]) / norm

        if self.goal2 is not None:
            norm = np.linalg.norm(x.squeeze() - self.goal2) + self.d
            # print(f'{norm=}')
            u_att_x += self.Katt*(x[0] - self.goal2[0]) / norm
            u_att_y += self.Katt*(x[1] - self.goal2[1]) / norm

        u_rep_x = 0
        u_rep_y = 0
        for obs in self.obstacles:
            rho_x = np.linalg.norm(x.squeeze() - obs.squeeze()) - self.d_obs
            # if rho_x < 1e-3:
            #     rho_x = 1e-3
            if rho_x <= self.rho_0 or self.disable_rho0:
                if self.gain_fn is None:
                    # gain = (self.Krep/rho_x**3) * (1 - rho_x/self.rho_0)
                    # gain = -self.Krep*(self.rho_0 - rho_x) / (self.rho_0*rho_x**4)
                    gain = self.Krep*(self.rho_0 - rho_x) / (self.rho_0*rho_x**3)
                    u_rep_x += gain*(x[0] - obs[0])
                    u_rep_y += gain*(x[1] - obs[1])
                else:
                    u_rep = self.gain_fn(x, obs, rho_x, self.rho_0)
                    u_rep_x += u_rep[0]
                    u_rep_y += u_rep[1]

        # if np.abs(u_rep_x) > 50 or np.abs(u_rep_y) > 50:
        #     print(f'{u_rep_x=}\n{u_rep_y=}')
        # if u_rep_x > 100:
        #     u_rep_x = 100
        # elif u_rep_x < -100:
        #     u_rep_x = -100

        # if u_rep_y > 100:
        #     u_rep_y = 100
        # elif u_rep_y < -100:
        #     u_rep_y = -100

        return np.array([-u_att_x + u_rep_x,
                         -u_att_y + u_rep_y])

    def event(self, t, x):
        distance = np.linalg.norm(x - self.goal)

        if distance < 1e-3:
            distance = 0.0

        return distance

    event.terminal = True  # Required for the integration to stop early


if __name__ == '__main__':
    plt.close('all')
    seed = 3
    rng = default_rng(seed)

    n = 10
    d_obs = 0.5
    goal = np.array([100, 0], dtype=float)
    x0 = np.array([0.1, 0.1], dtype=float)
    obstacles = [np.array([1, 2], dtype=float),  # Obstacle positions (m)
                 np.array([2.5, 3], dtype=float)]
    obstacles = [np.array([8, 0], dtype=float),  # Obstacle positions (m)
                  np.array([20, 2], dtype=float),
                  np.array([60, 1], dtype=float),
                  np.array([40, 2], dtype=float),
                  np.array([50, -3], dtype=float),
                  np.array([80, -3], dtype=float),
                  np.array([30, -1], dtype=float)]

    fapf = FAPF(x0, obstacles, goal, Katt=10, Krep=1, rho_0=0.1, d_obs=0.5)
    res = solve_ivp(fapf.fn, (0, 60), x0, method='LSODA', max_step=1, events=fapf.event)
    plt.figure()
    plt.plot(res.y[0], res.y[1])

    # traj = generate_apf_trajectory(x0, goal, obstacles, n=n)
    # traj.plot(showCpts=True)

    trajs = []
    for i in range(100):
        print(i)
        trajs.append(generate_apf_trajectory(x0, goal, obstacles, n=n, d_obs=d_obs, rho_std=0.1, tf_max=600, rng=rng))

    fig, ax = plt.subplots()
    for traj in trajs:
        traj.plot(ax, showCpts=False)

    for obs in obstacles:
        artist = Circle(obs, radius=d_obs)
        ax.add_artist(artist)
