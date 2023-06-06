#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:04:59 2023

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
from geometry_msgs.msg import TwistStamped
from jellyfish_search_msgs.msg import BernsteinTrajectory
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64


#TODO
class ThrusterController(Node):
    def __init__(self):
        super().__init__('thruster_controller')

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
                ('robot_frame_id', 'wamv')
                ]
            )

        self.thruster_left_pub = self.create_publisher(Float64, 'wamv/thrusters/left/thrust', 10)
        self.thruster_right_pub = self.create_publisher(Float64, 'wamv/thrusters/right/thrust', 10)

        self.odom_sub = self.create_subscription(Odometry, 'wamv/sensors/position/ground_truth_odometry',
                                                 self.odom_cb, 10)
        self.cmd_vel_sub = self.create_subscription(TwistStamped, 'cmd_vel', self.cmd_vel_cb, 10)

        self.speed = None
        self.angular_rate = None

        self.v_err_int = 0.0
        self.w_err_int = 0.0

        self.v_err_last = 0.0
        self.w_err_last = 0.0

        self.t_last = self.get_clock().now()

    def odom_cb(self, data: Odometry):
        self.speed = data.twist.twist.linear.x
        self.angular_rate = data.twist.twist.angular.z


    def cmd_vel_cb(self, data: TwistStamped):
        t = Time.from_msg(data.header.stamp)
        dt = (t - self.t_last).nanoseconds*1e-9
        self.t_last = t
        v_ref = data.twist.linear.x
        w_ref = data.twist.angular.z

        v_err = v_ref - self.speed
        w_err = w_ref - self.angular_rate

        kp_v = self.get_parameter('gain_pos_p').value
        ki_v = self.get_parameter('gain_pos_i').value
        kp_w = self.get_parameter('gain_theta_p').value
        ki_w = self.get_parameter('gain_theta_i').value

        v_cmd = v_ref #+ kp_v*v_err + ki_v*self.v_err_int
        w_cmd = w_ref #+ kp_w*w_err + ki_w*self.w_err_int

        thrust_right = 200*(v_cmd + w_cmd)
        thrust_left = 200*(v_cmd - w_cmd)

        # Clip the thrusts and also avoid integrator windup
        if thrust_right > 400 or thrust_right < -400 or thrust_left > 400 or thrust_right < -400:
            thrust_right = _clip(thrust_right, -400., 400.)
            thrust_left = _clip(thrust_left, -400., 400.)
        else:
            self.v_err_int += v_err*dt
            self.w_err_int += w_err*dt

        self.thruster_left_pub.publish(Float64(data=thrust_left))
        self.thruster_right_pub.publish(Float64(data=thrust_right))


def _clip(val, min_val, max_val):
    if val < min_val:
        val = min_val
    elif val > max_val:
        val = max_val

    return val


def main(args=None):
    rclpy.init(args=args)
    thruster_controller = ThrusterController()

    try:
        rclpy.spin(thruster_controller)
    except KeyboardInterrupt:
        thruster_controller.get_logger().info('Keyboard interrupt. Quitting.')
    finally:
        # Make sure we tell the boat to stop before turning off the controller
        thruster_controller.thruster_left_pub.publish(Float64(data=0.0))
        thruster_controller.thruster_right_pub.publish(Float64(data=0.0))
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        thruster_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
