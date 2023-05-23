#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 01:47:41 2023

@author: magicbycalvin
"""
import json
import os
import xml.etree.ElementTree as ET

import numpy as np


OBSTACLE_SAFE_DIST = 5.0  # m


if __name__ == '__main__':
    vrx_gz_path = '/home/magicbycalvin/vrx_ws/src/vrx/vrx_gz'
    model_name = 'short_navigation_course_2'
    fname = os.path.join(vrx_gz_path, 'models', model_name, 'model.sdf')
    tree = ET.parse(fname)
    root = tree.getroot()
    model = tree.find('model')

    obstacles = np.array([list(map(float, i.find('pose').text.split())) for i in model
                          if 'bound' in i.find('name').text or 'obstacle' in i.find('name').text])
    obstacles = obstacles[:, :3].T

    translation = np.array([-524, 198, 0])[:, np.newaxis]
    theta = -1.44
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])

    obstacles_map = R@obstacles + translation
    obstacles_list = obstacles_map.T.tolist()
    [i.append(OBSTACLE_SAFE_DIST) for i in obstacles_list]

    print(obstacles_list)

    with open('../config/'+model_name+'.json', 'w') as f:
        json.dump(obstacles_list, f)
