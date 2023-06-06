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
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from BeBOT.polynomial.bernstein import Bernstein
from geometry_msgs.msg import PoseStamped, TwistStamped
from jellyfish_search_msgs.msg import BernsteinTrajectory
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64


class TrajectoryTracker(Node):
    def __init__(self):
        super().__init__('trajectory_tracker')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('gain_pos_p', 1.0),
                # ('gain_pos_i', 1.0),
                # ('gain_pos_v', 1.0),
                ('gain_psi_p', 1.0),
                # ('gain_psi_i', 1.0),
                # ('gain_psi_d', 1.0),
                ('odom_topic', 'odometry'),
                ('cmd_vel_topic', 'cmd_vel'),
                ('world_frame_id', 'world'),
                ('robot_frame_id', 'base_link'),
                ('phase_unwrap_heading', True)
                ]
            )

        self.cmd_vel_pub = self.create_publisher(TwistStamped, self.get_parameter('cmd_vel_topic').value, 10)
        self.pose_ref_pub = self.create_publisher(PoseStamped, 'pose_ref', 10)
        self.twist_ref_pub = self.create_publisher(TwistStamped, 'twist_ref', 10)
        # self.x_ref_pub = self.create_publisher(Float64, 'x_ref', 10)
        # self.y_ref_pub = self.create_publisher(Float64, 'y_ref', 10)
        # self.psi_ref_pub = self.create_publisher(Float64, 'psi_ref', 10)
        # self.speed_ref_pub = self.create_publisher(Float64, 'speed_ref', 10)
        # self.x_err_pub = self.create_publisher(Float64, 'x_err', 10)
        # self.y_err_pub = self.create_publisher(Float64, 'y_err', 10)
        # self.psi_err_pub = self.create_publisher(Float64, 'psi_err', 10)
        # self.speed_err_pub = self.create_publisher(Float64, 'speed_err', 10)

        self.odom_sub = self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self.odom_cb,
                                                 QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT))
        self.traj_sub = self.create_subscription(BernsteinTrajectory, 'bebot_trajectory', self.traj_cb, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.pose = None
        self.speed = None
        self.cur_traj = None
        self.cur_traj_msg_time = 0.0
        self.psi_last = 0.0
        self.t_last = self.get_clock().now()
        # self.x_err_int = 0.0
        # self.y_err_int = 0.0
        # self.v_err_int = 0.0
        # self.psi_err_int = 0.0
        # self.x_err_last = 0.0
        # self.y_err_last = 0.0
        # self.v_err_last = 0.0
        # self.psi_err_last = 0.0


        #TODO create parameter for timer period
        self.main_timer = self.create_timer(0.1, self.main_cb)

    def odom_cb(self, data: Odometry):
        r = R.from_quat([data.pose.pose.orientation.x,
                         data.pose.pose.orientation.y,
                         data.pose.pose.orientation.z,
                         data.pose.pose.orientation.w])
        _, _, psi = r.as_euler('xyz')

        if self.get_parameter('phase_unwrap_heading').value:
            if psi - self.psi_last > np.pi:
                psi += 2*np.pi
            elif psi - self.psi_last < np.pi:
                psi -= 2*np.pi

        self.pose = np.array([data.pose.pose.position.x,
                              data.pose.pose.position.y,
                              psi],
                             dtype=float)
        self.speed = data.twist.twist.linear.x


    def traj_cb(self, data: BernsteinTrajectory):
        self.cur_traj_msg_time = Time.from_msg(data.header.stamp)

        t0 = data.t0
        tf = data.tf
        cpts = np.array([(i.x, i.y) for i in data.cpts], dtype=float).T
        self.cur_traj = Bernstein(cpts=cpts, t0=t0, tf=tf)

        self.get_logger().info(f'Trajectory received: {self.cur_traj}')

    def main_cb(self):
        now = self.get_clock().now()

        if self.pose is None:
            self.get_logger().info('Waiting for pose.')
        elif self.cur_traj is None:
            self.get_logger().info('Waiting for trajectory.')
        else:
            kp_pos = self.get_parameter('gain_pos_p').value
            # ki_pos = self.get_parameter('gain_pos_i').value
            # kv_pos = self.get_parameter('gain_pos_v').value
            kp_psi = self.get_parameter('gain_psi_p').value
            # ki_psi = self.get_parameter('gain_psi_i').value
            # kd_psi = self.get_parameter('gain_psi_d').value

            t0 = self.cur_traj_msg_time
            t = (now - t0).nanoseconds*1e-9
            # dt = (now - self.t_last).nanoseconds*1e-9

            t = _clip(t, self.cur_traj.t0, self.cur_traj.tf)

            # x_veh = self.pose[0]
            # y_veh = self.pose[1]
            # psi_veh = self.pose[2]
            # v_veh = self.speed

            x_ref = float(self.cur_traj.x(t).squeeze())
            y_ref = float(self.cur_traj.y(t).squeeze())
            xdot_ref = float(self.cur_traj.diff().x(t).squeeze())
            ydot_ref = float(self.cur_traj.diff().y(t).squeeze())
            xddot_ref = float(self.cur_traj.diff().diff().x(t).squeeze())
            yddot_ref = float(self.cur_traj.diff().diff().y(t).squeeze())
            v_ref = float(np.linalg.norm((xdot_ref, ydot_ref)))
            psi_ref = float(np.arctan2(ydot_ref, xdot_ref))
            w_ref = float((yddot_ref*xdot_ref - xddot_ref*ydot_ref)/v_ref**2)

            pose_ref_msg = PoseStamped()
            pose_ref_msg.header.stamp = now.to_msg()
            pose_ref_msg.header.frame_id = self.get_parameter('world_frame_id').value
            pose_ref_msg.pose.position.x = x_ref
            pose_ref_msg.pose.position.y = y_ref
            r = R.from_euler('z', psi_ref)
            q = r.as_quat()
            pose_ref_msg.pose.orientation.x = q[0]
            pose_ref_msg.pose.orientation.y = q[1]
            pose_ref_msg.pose.orientation.z = q[2]
            pose_ref_msg.pose.orientation.w = q[3]

            twist_ref_msg = TwistStamped()
            twist_ref_msg.header.stamp = now.to_msg()
            twist_ref_msg.header.frame_id = self.get_parameter('robot_frame_id').value
            twist_ref_msg.twist.linear.x = v_ref
            twist_ref_msg.twist.angular.z = w_ref

            # x_err = np.cos(psi_veh)*(x_ref - x_veh) + np.sin(psi_veh)*(y_ref - y_veh)
            # y_err = -np.sin(psi_veh)*(x_ref - x_veh) + np.cos(psi_veh)*(y_ref - y_veh)
            # psi_err = psi_ref - psi_veh
            # v_err = v_ref - v_veh

            # d_x_err = (x_err - self.x_err_last)/dt
            # d_y_err = (y_err - self.y_err_last)/dt
            # d_v_err = (v_err - self.v_err_last)/dt
            # d_psi_err = (psi_err - self.psi_err_last)/dt

            # self.x_err_last = x_err
            # self.y_err_last = y_err
            # self.v_err_last = v_err
            # self.psi_err_last = psi_err

            # # Add the initial position of the robot to the ref values for easy debugging
            # self.x_ref_pub.publish(Float64(data=x_ref))
            # self.y_ref_pub.publish(Float64(data=y_ref))
            # self.psi_ref_pub.publish(Float64(data=psi_ref))
            # self.speed_ref_pub.publish(Float64(data=v_ref))
            # self.x_err_pub.publish(Float64(data=x_err))
            # self.y_err_pub.publish(Float64(data=y_err))
            # self.psi_err_pub.publish(Float64(data=psi_err))
            # self.speed_err_pub.publish(Float64(data=v_err))

            # v_cmd = v_ref + kp_pos*x_err + ki_pos*self.x_err_int + kv_pos*v_err
            # w_cmd = kp_psi*psi_err + ki_psi*self.psi_err_int
            # w_cmd = 0.5*kp_psi*(np.sin(psi_ref)*np.cos(psi_veh) - np.cos(psi_ref)*np.sin(psi_veh))

            # v_cmd = v_ref + kp_pos*x_err + ki_pos*self.x_err_int #+ kv_pos*v_err
            # w_cmd = kp_psi*y_err + ki_psi*self.y_err_int + kd_psi*d_psi_err

            veh_pos = self.pose[:2][:, np.newaxis]
            veh_psi = self.pose[2]
            des_pos = np.array([[x_ref],
                                [y_ref]])
            # v_cmd, w_cmd = tt_controller(veh_pos, veh_psi, des_pos, psi_ref, v_ref, w_ref, kp_pos, kp_psi)
            v_cmd, w_cmd = pf_controller(veh_pos, veh_psi, des_pos, kp_pos, kp_psi)

            cmd_vel_msg = TwistStamped()
            cmd_vel_msg.header.stamp = now.to_msg()
            cmd_vel_msg.header.frame_id = self.get_parameter('robot_frame_id').value
            cmd_vel_msg.twist.linear.x = v_cmd
            cmd_vel_msg.twist.angular.z = w_cmd

            self.cmd_vel_pub.publish(cmd_vel_msg)
            self.pose_ref_pub.publish(pose_ref_msg)
            self.twist_ref_pub.publish(twist_ref_msg)

        self.t_last = now


def _clip(val, min_val, max_val):
    if val < min_val:
        val = min_val
    elif val > max_val:
        val = max_val

    return val


def tt_controller(veh_pos, veh_psi, des_pos, des_psi, des_v, des_w, kp, kr):
    err_pos = des_pos - veh_pos

    RW_I = np.array([[np.cos(veh_psi), -np.sin(veh_psi)],
                     [np.sin(veh_psi), np.cos(veh_psi)]])
    RT_I = np.array([[np.cos(des_psi), -np.sin(des_psi)],
                     [np.sin(des_psi), np.cos(des_psi)]])

    b1 = RW_I[:, 0][:, np.newaxis]
    b2 = RW_I[:, 1][:, np.newaxis]
    b1t = RT_I[:, 0][:, np.newaxis]
    b2t = RT_I[:, 1][:, np.newaxis]

    b1d = kp*err_pos + des_v*b1t
    b1d = b1d / np.linalg.norm(b1d)
    b2d = np.array([[b1d[1, 0]],
                    [-b1d[0, 0]]])

    RD_I = np.concatenate((b1d, b2d), axis=1)
    RW_D = RD_I.T@RW_I

    err_r = -(1/2)*RW_D[0, 1]

    v = (kp*err_pos.T + des_v*b1t.T)@b1
    w = des_w - kr*err_r

    return float(v), float(w)


def pf_controller(veh_pos, veh_psi, des_pos, kp, kr):
    RW_I = np.array([[np.cos(veh_psi), -np.sin(veh_psi)],
                     [np.sin(veh_psi), np.cos(veh_psi)]])
    pos_err_w = RW_I.T@(des_pos - veh_pos)

    v = kp * pos_err_w[0, 0]
    w = kr * pos_err_w[1, 0]

    return float(v), float(w)


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
