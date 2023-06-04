#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:27:02 2023

@author: magicbycalvin
"""
import json

from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from nml_bag import Reader

from BeBOT.polynomial.bernstein import Bernstein
from plotting_utils import setRCParams, resetRCParams


def traj_msg_to_bern(traj_msg):
    t0 = traj_msg['t0']
    tf = traj_msg['tf']

    cpts = np.array([(i['x'], i['y']) for i in traj_msg['cpts']]).T

    traj = Bernstein(cpts, t0, tf)

    return traj


if __name__ == '__main__':
    ###
    # Trajectories
    ###
    fname = '/home/magicbycalvin/Desktop/rosbag2_2023_06_03-20_30_43/rosbag2_2023_06_03-20_30_43_0.db3'
    # topics = ['/bebot_trajectory',
    #           '/bebot_trajectory_array',
    #           '/goal',
    #           '/mavros/global_position/global',
    #           '/mavros/global_position/local',
    #           '/mavros/setpoint_velocity/cmd_vel',
    #           '/mavros/state',
    #           '/move_base_simple/goal',
    #           '/obstacles',
    #           '/speed_err',
    #           '/speed_ref',
    #           '/theta_err',
    #           '/theta_ref',
    #           '/x_err',
    #           '/x_ref',
    #           '/y_err',
    #           '/y_ref']
    # reader = Reader(filepath=fname, , topics=topics)
    reader = Reader(filepath=fname)
    t0 = next(reader)['time_ns']

    # x_err_reader = Reader(filepath=fname, topics=['/x_err'])
    # y_err_reader = Reader(filepath=fname, topics=['/y_err'])
    # x_ref_reader = Reader(filepath=fname, topics=['/x_ref'])
    # y_ref_reader = Reader(filepath=fname, topics=['/y_ref'])
    pose_ref_reader = Reader(filepath=fname, topics=['/pose_ref'])
    twist_ref_reader = Reader(filepath=fname, topics=['/twist_ref'])
    goal_reader = Reader(filepath=fname, topics=['/goal'])
    gps_reader = Reader(filepath=fname, topics=['/mavros/global_position/local'])
    state_reader = Reader(filepath=fname, topics=['/mavros/state'])
    traj_reader = Reader(filepath=fname, topics=['/bebot_trajectory'])
    traj_array_reader = Reader(filepath=fname, topics=['/bebot_trajectory_array'])

    # x_err = np.array([((i['time_ns']-t0)*1e-9, i['data']) for i in x_err_reader])
    # y_err = np.array([((i['time_ns']-t0)*1e-9, i['data']) for i in y_err_reader])
    pose_ref = [((i['time_ns'] - t0)*1e-9, i['pose']) for i in pose_ref_reader]
    t_ref = np.array([i[0] for i in pose_ref])
    x_ref = np.array([i[1]['position']['x'] for i in pose_ref])
    y_ref = np.array([i[1]['position']['y'] for i in pose_ref])
    psi_ref = np.array([R.from_quat([
        i[1]['orientation']['x'],
        i[1]['orientation']['y'],
        i[1]['orientation']['z'],
        i[1]['orientation']['w']]).as_euler('xyz')[2] for i in pose_ref])
    twist_ref = [((i['time_ns'] - t0)*1e-9, i['twist']) for i in twist_ref_reader]
    goal_pos = np.array([((i['time_ns']-t0)*1e-9,
                          i['pose']['position']['x'],
                          i['pose']['position']['y'])
                         for i in goal_reader])
    gps = np.array([((i['time_ns']-t0)*1e-9,
                     i['pose']['pose']['position']['x'],
                     i['pose']['pose']['position']['y'],
                     R.from_quat([
                         i['pose']['pose']['orientation']['x'],
                         i['pose']['pose']['orientation']['y'],
                         i['pose']['pose']['orientation']['z'],
                         i['pose']['pose']['orientation']['w']]).as_euler('xyz')[2])
                    for i in gps_reader])
    trajectories = [i for i in traj_reader]
    traj_arrays = [i for i in traj_array_reader]

    time_flag = True
    guided_times = []
    for record in state_reader:
        if time_flag and record['mode'] == 'GUIDED':
            time_flag = False
            guided_times.append([(record['time_ns']-t0)*1e-9])
        elif not time_flag and record['mode'] != 'GUIDED':
            time_flag = True
            guided_times[-1].append((record['time_ns']-t0)*1e-9)

    ###
    # Obstacles
    ###
    obstacle_fname = '/home/magicbycalvin/ros2_ws/src/jellyfish_search/config/final_obs_course.json'
    with open(obstacle_fname, 'r') as f:
        obstacles = json.load(f)

    ###
    # Plotting
    ###
    setRCParams()
    plt.close('all')
    fig1, ax1 = plt.subplots()
    ax1.plot(gps[:, 0], gps[:, 3], label='$\psi_\mathrm{gps}$')
    ax1.plot(t_ref, psi_ref, label='$\psi_\mathrm{ref}$')
    ax1.set_title('Heading Angle')
    ax1.legend()

    fig2, ax2 = plt.subplots()
    ax2.plot(t_ref, x_ref, label=r'$x_\mathrm{ref}$')
    ax2.plot(t_ref, y_ref, label=r'$y_\mathrm{ref}$')
    ax2.plot(gps[:, 0], gps[:, 1], label=r'$x_\mathrm{gps}$')
    ax2.plot(gps[:, 0], gps[:, 2], label=r'$y_\mathrm{gps}$')
    ax2.set_ylim([-30, 100])
    ax2.set_title('Reference and GPS Positions')
    ax2.legend()

    for ax in [ax1, ax2]:
        ax.set_xlim([min(min(guided_times)), max(max(guided_times))])
        for guided_time in guided_times:
            ax.axvspan(guided_time[0], guided_time[1], color='green', alpha=0.25)

    fig3, ax3 = plt.subplots()
    ax3.plot(gps[:, 1], gps[:, 2])
    for obs in obstacles:
        ax3.add_artist(Circle(obs[:2], radius=obs[-1]))
    for traj_msg in trajectories:
        traj = traj_msg_to_bern(traj_msg)
        traj.plot(ax3, showCpts=False, zorder=10)
    for traj_array in traj_arrays:
        for traj_msg in traj_array['trajectories']:
            traj = traj_msg_to_bern(traj_msg)
            traj.plot(ax3, showCpts=False, color='g', alpha=0.2, lw=1)
    ax3.set_xlim([-100, 100])
    ax3.set_ylim([-100, 100])

    resetRCParams()
