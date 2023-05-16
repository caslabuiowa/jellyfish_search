#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:50:48 2023

@author: magicbycalvin
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from BeBOT.polynomial.bernstein import Bernstein
from jellyfish_search_msgs.msg import BernsteinTrajectory
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64


class TrajectoryTracker(Node):
    def __init__(self):
        super().__init__('trajectory_tracker')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('gain_pos_p', 10.0),
                ('gain_pos_i', 1.0),
                ('gain_pos_v', 1.0),
                ('gain_theta_p', 1.0),
                ('gain_theta_i', 1.0),
                ('gain_theta_d', 1.0),
                ('world_frame_id', 'world'),
                ('robot_frame_id', 'base_link')
                ]
            )

        self.thruster_left_pub = self.create_publisher(Float64, 'wamv/thrusters/left/thrust', 10)
        self.thruster_right_pub = self.create_publisher(Float64, 'wamv/thrusters/right/thrust', 10)

        self.odom_sub = self.create_subscription(Odometry, 'odometry/local', self.odom_cb, 10)
        self.traj_sub = self.create_subscription(BernsteinTrajectory, 'bebot_trajectory', self.traj_cb, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.pose = None
        self.speed = None
        self.cur_traj = None
        self.cur_traj_msg_time = 0.0
        self.x_err_int = 0.0
        self.y_err_int = 0.0
        self.v_err_int = 0.0
        self.theta_err_int = 0.0
        self.t_last = self.get_clock().now()

        #TODO create parameter for timer period
        self.main_timer = self.create_timer(0.1, self.main_cb)

    def odom_cb(self, data: Odometry):
        r = R.from_quat([data.pose.pose.orientation.x,
                         data.pose.pose.orientation.y,
                         data.pose.pose.orientation.z,
                         data.pose.pose.orientation.w])
        _, _, theta = r.as_rotvec()
        self.pose = np.array([data.pose.pose.position.x,
                              data.pose.pose.position.y,
                              theta],
                             dtype=float)
        self.speed = data.twist.twist.linear.x


    def traj_cb(self, data: BernsteinTrajectory):
        self.cur_traj_msg_time = Time.from_msg(data.header.stamp)

        t0 = data.t0
        tf = data.tf
        cpts = np.array([(i.x, i.y) for i in data.cpts], dtype=float).T
        self.cur_traj = Bernstein(cpts=cpts, t0=t0, tf=tf)

        # Zero all integrator errors to mitigate integral windup
        self.x_err_int = 0.0
        self.y_err_int = 0.0
        self.v_err_int = 0.0
        self.theta_err_int = 0.0

    def main_cb(self):
        now = self.get_clock().now()

        if self.pose is None:
            self.get_logger().info('Waiting for pose.')
        elif self.cur_traj is None:
            self.get_logger().info('Waiting for trajectory.')
        else:
            kp_pos = self.get_parameter('gain_pos_p').value
            ki_pos = self.get_parameter('gain_pos_i').value
            kv_pos = self.get_parameter('gain_pos_v').value
            kp_theta = self.get_parameter('gain_theta_p').value
            ki_theta = self.get_parameter('gain_theta_i').value
            kd_theta = self.get_parameter('gain_theta_d').value

            t0 = self.cur_traj_msg_time
            t = (now - t0).nanoseconds*1e-9
            dt = (now - self.t_last).nanoseconds*1e-9

            x_veh = self.pose[0]
            y_veh = self.pose[1]
            theta_veh = self.pose[2]
            v_veh = self.speed

            x_ref = self.cur_traj.x(t).squeeze()
            y_ref = self.cur_traj.y(t).squeeze()
            xdot_ref = self.cur_traj.diff().x(t).squeeze()
            ydot_ref = self.cur_traj.diff().y(t).squeeze()
            v_ref = np.linalg.norm((xdot_ref, ydot_ref))
            theta_ref = np.arctan2(ydot_ref, xdot_ref)

            x_err = np.cos(theta_veh)*(x_ref - x_veh) + np.sin(theta_veh)*(y_ref - y_veh)
            y_err = -np.sin(theta_veh)*(x_ref - x_veh) + np.cos(theta_veh)*(y_ref - y_veh)
            theta_err = theta_ref - theta_veh
            v_err = v_ref - v_veh

            v_cmd = v_ref*np.cos(theta_err) + kp_pos*x_err + ki_pos*self.x_err_int + kv_pos*v_err
            w_cmd = kp_theta*theta_err + ki_theta*self.theta_err_int

            thrust_right = v_cmd + w_cmd
            thrust_left = v_cmd - w_cmd

            self.thruster_left_pub.publish(Float64(data=thrust_left))
            self.thruster_right_pub.publish(Float64(data=thrust_right))

            self.x_err_int += x_err*dt
            self.y_err_int += y_err*dt
            self.v_err_int += v_err*dt
            self.theta_err_int += theta_err*dt

        self.t_last = now


def main(args=None):
    rclpy.init(args=args)
    trajectory_tracker = TrajectoryTracker()

    try:
        rclpy.spin(trajectory_tracker)
    except KeyboardInterrupt:
        trajectory_tracker.get_logger().info('Keyboard interrupt. Quitting.')
    finally:
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        trajectory_tracker.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
