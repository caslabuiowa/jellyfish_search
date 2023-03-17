#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:17:45 2023

@author: magicbycalvin
"""
import csv
import os
import time
import sys
sys.path.append('/home/magicbycalvin/Projects/last_minute_comprehensive/BeBOT')

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np

from jfs_core.jellyfish_search import is_feasible, cost_fn, ProblemParameters, SolverParameters, project_goal
from jfs_core.cbf_search import generate_cbf_trajectory
from plotting_utils import save_all_figures, setRCParams, resetRCParams


if __name__ == '__main__':
    rng = np.random.default_rng(3)
    cbf_kwargs = {
        'n': 15,
        'Katt_std': 1,
        'Krep_std': 1,
        'rho_std': 1,
        'd_obs': 1,
        'tf_max': 60,
        't0': 0,
        'delta': 0.01,
        'rng': rng,
        'debug': True}
    vmax = 3
    wmax = 2*np.pi
    rsafe = 1000  # Ignoring this constraint for this example
    safe_dist = cbf_kwargs['d_obs']
    x0 = np.array([0, 0], dtype=float)
    goal = np.array([10, 10], dtype=float)
    obstacles = np.array([
        [5, 6],
        [1, 3],
        [6, 8]], dtype=float)

    trajs = []
    times = []
    for i in range(1000):
        tstart = time.time()
        trajs.append(generate_cbf_trajectory(x0, goal, obstacles, **cbf_kwargs))
        elapsed = time.time() - tstart
        print(f'Trajectory {i} took {elapsed} s')
        times.append(elapsed)

    fidx = 0
    fname = f'cbf_jfs_data_{fidx}.csv'
    while os.path.isfile(fname):
        fidx += 1
        fname = f'cbf_jfs_data_{fidx}.csv'

    with open(fname, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Computation Time', 'Katt', 'Krep', 'rho_0', 't0', 'tf', 'cpts'])
        for i, traj_tuple in enumerate(trajs):
            traj, Katt, Krep, rho_0, result = traj_tuple
            comp_time = times[i]
            csv_writer.writerow([comp_time, Katt, Krep, rho_0, traj.t0, traj.tf, traj.cpts])

    plt.close('all')
    setRCParams()
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    for i, traj_tuple in enumerate(trajs):
        traj = traj_tuple[0]
        if is_feasible(traj, obstacles, safe_dist, vmax, wmax, rsafe):
            color = 'g'
            traj.plot(ax2, showCpts=False)
        else:
            color = 'r'
        if times[i] > 1:
            line_style = '--'
        else:
            line_style = '-'

        traj.plot(ax, showCpts=False, c=color, ls=line_style, alpha=0.3)

    for obs in obstacles:
        ax.add_artist(Circle(obs, radius=safe_dist))
        ax2.add_artist(Circle(obs, radius=safe_dist))

    ax2.plot(goal[0], goal[1], 'r*', ms=30)

    save_all_figures([f'all_trajectories_{fidx}', f'feasible_trajectories_{fidx}'])
    resetRCParams()
