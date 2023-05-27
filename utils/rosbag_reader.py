#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:27:02 2023

@author: magicbycalvin
"""
import matplotlib.pyplot as plt
import numpy as np

from nml_bag import Reader

from plotting_utils import setRCParams, resetRCParams


if __name__ == '__main__':
    fname = '/home/magicbycalvin/Desktop/rosbag2_2023_05_25-15_06_38/rosbag2_2023_05_25-15_06_38_0.db3'
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

    x_err_reader = Reader(filepath=fname, topics=['/x_err'])
    y_err_reader = Reader(filepath=fname, topics=['/y_err'])
    x_ref_reader = Reader(filepath=fname, topics=['/x_ref'])
    y_ref_reader = Reader(filepath=fname, topics=['/y_ref'])
    goal_reader = Reader(filepath=fname, topics=['/goal'])
    gps_reader = Reader(filepath=fname, topics=['/mavros/global_position/local'])
    state_reader = Reader(filepath=fname, topics=['/mavros/state'])

    x_err = np.array([((i['time_ns']-t0)*1e-9, i['data']) for i in x_err_reader])
    y_err = np.array([((i['time_ns']-t0)*1e-9, i['data']) for i in y_err_reader])
    x_ref = np.array([((i['time_ns']-t0)*1e-9, i['data']) for i in x_ref_reader])
    y_ref = np.array([((i['time_ns']-t0)*1e-9, i['data']) for i in y_ref_reader])
    goal_pos = np.array([((i['time_ns']-t0)*1e-9,
                          i['pose']['position']['x'],
                          i['pose']['position']['y'])
                         for i in goal_reader])
    gps = np.array([((i['time_ns']-t0)*1e-9,
                     i['pose']['pose']['position']['x'],
                     i['pose']['pose']['position']['y'])
                    for i in gps_reader])

    time_flag = True
    guided_times = []
    for record in state_reader:
        if time_flag and record['mode'] == 'GUIDED':
            time_flag = False
            guided_times.append([(record['time_ns']-t0)*1e-9])
        elif not time_flag and record['mode'] != 'GUIDED':
            time_flag = True
            guided_times[-1].append((record['time_ns']-t0)*1e-9)

    setRCParams()
    plt.close('all')
    fig1, ax1 = plt.subplots()
    ax1.plot(x_err[:, 0], x_err[:, 1], label='$x_\mathrm{err}$')
    ax1.plot(y_err[:, 0], y_err[:, 1], label='$y_\mathrm{err}$')
    ax1.set_title('Position Tracking Error')
    ax1.legend()

    fig2, ax2 = plt.subplots()
    ax2.plot(x_ref[:, 0], x_ref[:, 1], label='$x_\mathrm{ref}$')
    ax2.plot(y_ref[:, 0], y_ref[:, 1], label='$y_\mathrm{ref}$')
    ax2.plot(gps[:, 0], gps[:, 1], label='$x_\mathrm{gps}$')
    ax2.plot(gps[:, 0], gps[:, 2], label='$y_\mathrm{gps}$')
    ax2.set_ylim([-30, 100])
    ax2.set_title('Reference and GPS Positions')
    ax2.legend()

    fig3, ax3 = plt.subplots()
    ax3.plot(gps[:, 0], gps[:, 1])
    ax3.plot(gps[:, 0], gps[:, 2])
    ax3.set_ylim([-30, 101])

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim([400, 800])
        for guided_time in guided_times:
            ax.axvspan(guided_time[0], guided_time[1], color='green', alpha=0.25)

    resetRCParams()
