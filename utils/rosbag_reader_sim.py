#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:27:02 2023

@author: magicbycalvin
"""
import json

import matplotlib as mpl
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np

from nml_bag import Reader

from BeBOT.polynomial.bernstein import Bernstein
from plotting_utils import setRCParams, resetRCParams


def traj_msg_to_bern(traj_msg):
    t0 = traj_msg['t0']
    tf = traj_msg['tf']

    try:
        cpts = np.array([(i['x'], i['y']) for i in traj_msg['cpts']]).T
    except IndexError:
        cpts = np.array([(i['x']) for i in traj_msg['cpts']]).T

    traj = Bernstein(cpts, t0, tf)

    return traj


if __name__ == '__main__':
    ###
    # Trajectories
    ###
    # fname = '/home/magicbycalvin/Desktop/JFS_testing/rosbag2_2023_05_31-12_19_34/rosbag2_2023_05_31-12_19_34_0.db3'
    fname = '/home/magicbycalvin/Desktop/JFS_testing/rosbag2_2023_05_31-13_42_40/rosbag2_2023_05_31-13_42_40_0.db3'

    reader = Reader(filepath=fname)
    t0 = next(reader)['time_ns']

    goal_reader = Reader(filepath=fname, topics=['/goal'])
    gps_reader = Reader(filepath=fname, topics=['/mavros/global_position/local'])
    traj_reader = Reader(filepath=fname, topics=['/bebot_trajectory'])
    traj_array_reader = Reader(filepath=fname, topics=['/bebot_trajectory_array'])
    obstacle_distances = Reader(filepath=fname, topics=['/obstacle_distances'])

    goal_pos = np.array([((i['time_ns']-t0)*1e-9,
                          i['pose']['position']['x'],
                          i['pose']['position']['y'])
                         for i in goal_reader])
    gps = np.array([((i['time_ns']-t0)*1e-9,
                     i['pose']['pose']['position']['x'],
                     i['pose']['pose']['position']['y'])
                    for i in gps_reader])
    trajectories = [i for i in traj_reader]
    traj_arrays = [i for i in traj_array_reader]
    obstacle_distances = [i for i in obstacle_distances]


    ###
    # Obstacles
    ###
    obstacle_fname = '/home/magicbycalvin/ros2_ws/src/jellyfish_search/config/obstacles.json'
    with open(obstacle_fname, 'r') as f:
        obstacles = json.load(f)

    ###
    # Plotting
    ###
    setRCParams()
    plt.close('all')
    fig1, ax1 = plt.subplots()
    for dists in obstacle_distances:
        for dist in dists['trajectories']:
            t0 = dist['t0']
            tf = dist['tf']
            cpts = np.atleast_2d([i['x'] for i in dist['cpts']])
            dist_traj = Bernstein(cpts, t0, tf)
            dist_traj.plot(ax1, showCpts=False)
    ax1.set_ylim([-1, 10])

    # fig2, ax2 = plt.subplots()
    # ax2.set_title('Reference and GPS Positions')
    # ax2.legend()

    fig3, ax3 = plt.subplots()
    # ax3.plot(gps[:, 1], gps[:, 2])
    ntrajs = len(trajectories)
    cmap = mpl.colormaps['jet']
    for obs in obstacles:
        ax3.add_artist(Circle(obs[:2], radius=obs[-1]))
    for i, traj_msg in enumerate(trajectories):
        traj = traj_msg_to_bern(traj_msg)
        traj.plot(ax3, showCpts=False, zorder=10, color=cmap(i/(ntrajs-1)))
    for traj_array in traj_arrays:
        for traj_msg in traj_array['trajectories']:
            traj = traj_msg_to_bern(traj_msg)
            traj.plot(ax3, showCpts=False, color='g', alpha=0.2, lw=1)
    ax3.set_xlim([0, 170])
    ax3.set_ylim([-75, 75])

    resetRCParams()
