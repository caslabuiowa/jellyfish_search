#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 12:49:28 2023

@author: magicbycalvin
"""
import time

from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np

from jfs_core.jfs import JellyfishSearch

if __name__ == '__main__':
    num_trajectories = 100
    solver_params = dict(n_steps=100,
                         tf_max=60,
                         Katt=1,
                         Krep=100,
                         rho_0=10,
                         delta=0.0)

    goal_pos_std = 6.0
    obs_pos_std = 0#0.5
    obs_size_std = 0.3

    goal = np.array([20, 20], dtype=float)
    x0 = np.array([0.1, 0.1], dtype=float)
    obstacles = [np.array([8.5, 8.5], dtype=float),  # Obstacle positions (m)
                 np.array([1, 4], dtype=float),
                 np.array([7, 1], dtype=float),
                 np.array([15, 15], dtype=float),
                 np.array([50, -3], dtype=float),
                 np.array([80, -3], dtype=float),
                 np.array([30, -1], dtype=float)
                 ]
    obstacle_safe_distances = np.array([1.5, #6, #1
                                        2,
                                        3, #5, #3
                                        3,
                                        1,
                                        3,
                                        1], dtype=float)

    jfs = JellyfishSearch(rng_seed=1, num_workers=12)

    t_start = time.time()
    results = []
    for i in range(100):
        x0[1] += 0.1
        results += jfs.generate_trajectories(x0, goal, obstacles, obstacle_safe_distances, num_trajectories,
                                             obs_pos_std, obs_size_std, goal_pos_std)
    print(f'Elapsed time: {time.time() - t_start}')

    del jfs

    plt.close('all')
    fig, ax = plt.subplots()
    for res in results:
        ax.plot(res[0][:, 0], res[0][:, 1])

    for i, obs in enumerate(obstacles):
        ax.add_artist(Circle(obs, radius=obstacle_safe_distances[i]))
