#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 11:31:31 2023

@author: magicbycalvin
"""
import json

from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    fname = '/home/magicbycalvin/ros2_ws/src/jellyfish_search/utils/better_obs_course.json'

    with open(fname, 'r') as f:
        obstacles = np.array(json.load(f))

    plt.close('all')
    fig, ax = plt.subplots()
    for obs in obstacles:
        ax.add_artist(Circle(obs[:2], radius=obs[-1]))
    ax.set_xlim([obstacles[:, 0].min()-10, obstacles[:, 0].max()+10])
    ax.set_ylim([obstacles[:, 1].min()-10, obstacles[:, 1].max()+10])
