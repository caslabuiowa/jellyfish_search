#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:10:33 2023

@author: magicbycalvin
"""
import os
import time
import sys
sys.path.append('/home/magicbycalvin/Projects/last_minute_comprehensive/BeBOT')

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np

from jfs_core.jellyfish_search import generate_jellyfish_trajectories, generate_jellyfish_trajectories_timeout
from jfs_core.jellyfish_search import ProblemParameters, SolverParameters, project_goal


def plot_trajectories(ax, traj_family, n_trajs=10):
    for traj in traj_family[1:n_trajs]:
        traj[0].plot(ax, showCpts=False, color='g')
    traj_family[0][0].plot(ax, showCpts=False, color='b', lw=2)


def create_animation():
    seed = 3
    x0_std = 1

    n = 5
    t0 = 0
    tf = 30
    ndim = 2
    n_trajectories = 30

    planning_timestep = 3  # Seconds

    safe_planning_radius = 11
    safe_distance = 1
    maximum_speed = 3
    maximum_angular_rate = np.pi/4

    obstacles = [np.array([8.5, 0], dtype=float),  # Obstacle positions (m)
                 np.array([20, 2], dtype=float),
                 np.array([60, 1], dtype=float),
                 np.array([40, 2], dtype=float),
                 np.array([50, -3], dtype=float),
                 np.array([80, -3], dtype=float),
                 np.array([30, -1], dtype=float)]

    goal = np.array([60, 3], dtype=float)
    initial_position = np.array([0, 0.1], dtype=float)

    cur_goal = project_goal(initial_position, safe_planning_radius-1, goal)
    x0 = np.array([initial_position[0],
                    initial_position[1],
                    ], dtype=float)

    solver_params = SolverParameters('apf',
                                     {'rho_std': 0.1,
                                      'Katt_std': 1,
                                      'Krep_std': 1,
                                      'tf_max': 60},
                                     n_trajectories,
                                     rng_seed=seed)
    problem_params = ProblemParameters(n,
                                       ndim,
                                       x0,
                                       cur_goal,
                                       tf,
                                       obstacles,
                                       safe_distance,
                                       maximum_speed,
                                       maximum_angular_rate,
                                       safe_planning_radius)

    traj_family = generate_jellyfish_trajectories(problem_params, solver_params)
    x = traj_family[0][0](0)

    fig, ax = plt.subplots()
    line = ax.plot(x[0], x[1], 'r*', ms=5, zorder=10)
    for obs in obstacles:
        artist = Circle(obs, radius=safe_distance)
        ax.add_artist(artist)

    plot_trajectories(ax, traj_family)
    my_dpi = 192
    fig.set_figwidth(1920/my_dpi)
    fig.set_figheight(1152/my_dpi)
    ax.set_xlim([initial_position[0]-3, goal[0]+3])
    ax.set_ylim([initial_position[1]-5, goal[1]+5])
    plt.tight_layout()

    def update(frame):
        t = frame*0.033

        tlast = 0.0
        flag = 0
        sim_time = np.linspace(0, 15, 1001)

        x = update.traj_family[0][0](t-update.tlast)
        line[0].set_xdata([x[0]])
        line[0].set_ydata([x[1]])
        if t - update.tlast >= planning_timestep:
            print(f'{t=}')
            print(f'{update.tlast=}')
            cur_goal = project_goal(x, safe_planning_radius-1, goal)

            problem_params.x0 = x.squeeze()
            problem_params.goal = cur_goal

            update.traj_family = generate_jellyfish_trajectories(problem_params, solver_params)
            plot_trajectories(ax, update.traj_family)
            update.tlast = t

    update.tlast = 0
    update.traj_family = traj_family

    fa = FuncAnimation(fig=fig, func=update, frames=int(tf*33), interval=33)
    fidx = 0
    fname = f'animation_{fidx}.mp4'
    while os.path.isfile(fname):
        fidx += 1
        fname = f'animation_{fidx}.mp4'

    print(f'Saving to {fname}...')
    fa.save(fname, dpi=my_dpi)
    print('Done!')


if __name__ == '__main__':
    plt.close('all')
    create_animation()

if __name__ == '__main__' and False:
    seed = 3
    x0_std = 1

    n = 5
    t0 = 0
    tf = 30
    ndim = 2
    n_trajectories = 30

    planning_timestep = 3  # Seconds

    safe_planning_radius = 11
    safe_distance = 1
    maximum_speed = 3
    maximum_angular_rate = np.pi/4

    obstacles = [np.array([8.5, 0], dtype=float),  # Obstacle positions (m)
                 np.array([20, 2], dtype=float),
                 np.array([60, 1], dtype=float),
                 np.array([40, 2], dtype=float),
                 np.array([50, -3], dtype=float),
                 np.array([80, -3], dtype=float),
                 np.array([30, -1], dtype=float)]

    goal = np.array([60, 3], dtype=float)
    initial_position = np.array([0, 0.1], dtype=float)

    cur_goal = project_goal(initial_position, safe_planning_radius-1, goal)
    x0 = np.array([initial_position[0],
                    initial_position[1],
                    ], dtype=float)

    solver_params = SolverParameters('apf',
                                     {'rho_std': 0.1,
                                      'Katt_std': 1,
                                      'Krep_std': 1,
                                      'tf_max': 60},
                                     n_trajectories,
                                     rng_seed=seed)
    problem_params = ProblemParameters(n,
                                       ndim,
                                       x0,
                                       cur_goal,
                                       tf,
                                       obstacles,
                                       safe_distance,
                                       maximum_speed,
                                       maximum_angular_rate,
                                       safe_planning_radius)

    tstart = time.time()
    traj_family = generate_jellyfish_trajectories(problem_params, solver_params)
    print(f'Elapsed time for family of {n_trajectories} trajectories: {time.time() - tstart} s')

    plt.close('all')
    fig, ax = plt.subplots()
    for obs in obstacles:
        artist = Circle(obs, radius=safe_distance)
        ax.add_artist(artist)

    plot_trajectories(ax, traj_family)

    tlast = 0.0
    flag = 0
    sim_time = np.linspace(0, 15, 1001)
    for t in sim_time:
        if t - tlast >= planning_timestep:
            x = traj_family[0][0](t-tlast)
            cur_goal = project_goal(x, safe_planning_radius-1, goal)

            problem_params.x0 = x.squeeze()
            problem_params.goal = cur_goal

            tstart = time.time()
            traj_family = generate_jellyfish_trajectories(problem_params, solver_params)
            print(f'Elapsed time for family of {n_trajectories} trajectories: {time.time() - tstart} s')
            plot_trajectories(ax, traj_family)

            print(f'{t=}')
            print(f'{x=}')
            tlast = t
    print(f'Elapsed time: {time.time() - tstart}')
