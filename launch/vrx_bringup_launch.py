#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 11:16:22 2023

@author: magicbycalvin
"""
import os

from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
# from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
# from launch_ros.substitutions import FindPackageShare

from launch_xml.launch_description_sources import XMLLaunchDescriptionSource


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('jellyfish_search'),
        'config',
        'vrx_params.yaml'
        )

    rosbag_recorder = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-a'],
        output='screen'
        )

    jfs_controller_node = Node(
        package='jellyfish_search',
        namespace='',
        executable='jellyfish_controller',
        name='trajectory_generator',
        parameters=[config]
        )

    jfs_tt_node = Node(
        package='jellyfish_search',
        namespace='',
        executable='trajectory_tracker',
        name='trajectory_tracker',
        parameters=[config]
        )

    jfs_obstacle_node = Node(
        package='jellyfish_search',
        namespace='',
        executable='obstacle_detector',
        name='obstacle_detector',
        parameters=[config]
        )

    vrx_thrust_controller_node = Node(
        package='jellyfish_search',
        namespace='',
        executable='thruster_controller',
        name='thruster_controller',
        parameters=[config]
        )

    return LaunchDescription([
        jfs_controller_node,
        jfs_tt_node,
        jfs_obstacle_node,
        vrx_thrust_controller_node,
        rosbag_recorder
    ])
