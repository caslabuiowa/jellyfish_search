#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:00:29 2023

@author: magicbycalvin
"""
import json
import os

import numpy as np

from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.node import Node

from jellyfish_search_msgs.msg import Obstacle, ObstacleArray


class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')

        # Parameters
        obs_pub_freq_description = ParameterDescriptor(
            description='Frequency at which to publish the array of obstacle locations in Hz.')
        obs_frame_description = ParameterDescriptor(
            description='Frame ID for the obstacle array.')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('obstacle_publish_frequency', 10, obs_pub_freq_description),
                ('obstacle_frame_id', 'odom', obs_frame_description),
                ('obstacle_file', 'obstacles.json')
                ],
            )

        self.obstacle_pub = self.create_publisher(ObstacleArray, 'obstacles', 10)

        fname = os.path.join(get_package_share_directory('jellyfish_search'), 'config',
                             self.get_parameter('obstacle_file').value)

        self.get_logger().info(f'Loading obstacles from file: {fname}')
        with open(fname, 'r') as f:
            obstacles_list = json.load(f)
            self.static_obstacles = np.array(obstacles_list)

        self.get_logger().info(f'Static Obstacles:\n{self.static_obstacles}')

        obs_pub_freq = self.get_parameter('obstacle_publish_frequency').value
        self.obstacle_timer = self.create_timer(1/obs_pub_freq, self.obstacle_cb)

    def obstacle_cb(self):
        obs_array_msg = ObstacleArray()
        for obs in self.static_obstacles:
            obs_msg = Obstacle()
            obs_msg.header.frame_id = self.get_parameter('obstacle_frame_id').value
            obs_msg.header.stamp = self.get_clock().now().to_msg()
            obs_msg.position.x = obs[0]
            obs_msg.position.y = obs[1]
            obs_msg.position.z = obs[2]
            obs_msg.radius = obs[3]

            obs_array_msg.obstacles.append(obs_msg)

        self.obstacle_pub.publish(obs_array_msg)


def main(args=None):
    rclpy.init(args=args)
    obs_detector = ObstacleDetector()
    rclpy.spin(obs_detector)
    obs_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
