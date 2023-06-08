#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 12:36:58 2023

@author: magicbycalvin
"""
import json

from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc


def save_obstacles(obs_positions: np.ndarray, obs_sizes: np.ndarray, fname: str):

    obstacles = np.hstack((obs_positions, [[0]]*len(obs_positions), obs_sizes[:, np.newaxis])).tolist()
    obstacles.sort()

    with open(fname, 'w') as f:
        json.dump(obstacles, f)

if __name__ == '__main__':
    x_min = -30   # (m)
    x_max = 200    # (m)
    y_min = -200   # (m)
    y_max = 30    # (m)
    obs_rad_min = 2  # (m)
    obs_rad_max = 7  # (m)

    rng = np.random.default_rng(2)
    sampler = qmc.Sobol(d=2)
    obs_positions = sampler.random_base2(7)

    obs_positions[:, 0] = obs_positions[:, 0]*(x_max - x_min) + x_min
    obs_positions[:, 1] = obs_positions[:, 1]*(y_max - y_min) + y_min

    obs_tmp = []
    for obs in obs_positions:
        if obs[0] < 30 and obs[1] > -50:
            pass
        else:
            obs_tmp.append(obs)
    obs_positions = np.array(obs_tmp)

    obs_sizes = rng.integers(obs_rad_min, obs_rad_max+1, len(obs_positions))

    plt.close('all')
    fig, ax = plt.subplots()
    for obs, rad in zip(obs_positions, obs_sizes):
        ax.add_artist(Circle(obs, radius=rad))
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
