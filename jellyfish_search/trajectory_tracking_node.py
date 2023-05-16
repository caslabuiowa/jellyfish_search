#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:50:48 2023

@author: magicbycalvin
"""
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from BeBOT.polynomial.bernstein import Bernstein
from jellyfish_search_msgs.msg import BernsteinTrajectory


class TrajectoryTracker(Node):
    def __init__(self):
        super().__init__('trajectory_tracker')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('gain_p', 1.0),
                ('gain_i', 1.0),
                ('gain_d', 1.0),
                ('world_frame_id', 'world'),
                ('robot_frame_id', 'base_link')
                ]
            )

        self.traj_sub = self.create_subscription(BernsteinTrajectory, 'bebot_trajectory', self.traj_cb, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.cur_traj = None
        self.cur_traj_msg_time = 0.0

    def traj_cb(self, data: BernsteinTrajectory):
        sec, nsec = Time.from_msg(data.header.stamp).seconds_nanoseconds()
        self.cur_traj_msg_time = sec + nsec*1e-9

        t0 = data.t0
        tf = data.tf
        cpts = np.array([(i.x, i.y) for i in data.cpts], dtype=float).T
        self.cur_traj = Bernstein(cpts=cpts, t0=t0, tf=tf)


class Controller(Node):
    def __init__(self, Kp=0, Ki=0, Kd=0, Kangle=0, Kpw=0, bounds=(-1,1)):
        """
        """
        self._kp = Kp
        self._ki = Ki
        self._kd = Kd
        self._xe_int = 0
        self._ye_int = 0
        self._kangle = Kangle
        self._kpw = Kpw

        self.world_frame_id = 'world'
        self.robot_frame_id = 'AR_{}'.format(rospy.get_namespace()[-2])

        self._cmd_pose = PoseStamped()
        self._cmd_pose.header.frame_id = 'world'
        self._cmd_twist = TwistStamped()
        self._cmd_twist.header.frame_id = 'world'

#        self.droneName = 'AR_{}'.format(rospy.get_namespace()[-2])

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._sub_cmd_pose = rospy.Subscriber('cmd_pose',
                                              PoseStamped,
                                              self._updateCmdPose)
        self._sub_cmd_twist = rospy.Subscriber('cmd_twist',
                                               TwistStamped,
                                               self._updateCmdTwist)

#        self.debugPub = rospy.Publisher('debug', TwistStamped, queue_size=10)

        self._last_time = rospy.Time.now()

        self._clamp = lambda n: max(min(bounds[1], n), bounds[0])

    def _updateCmdPose(self, data):
        """
        Assumes the yaw angle is given in the z value of orientation, x, y,
        and w are unused.
        """
        self._cmd_pose = data
        self._cmd_pose.header.frame_id = self.world_frame_id

    def _updateCmdTwist(self, data):
        """

        """
        self._cmd_twist = data
        self._cmd_twist.header.frame_id = self.world_frame_id

    def init_pose(self, topic):
        self._cmd_pose = rospy.wait_for_message(topic, PoseStamped)

    def update(self, ardrone):
        """
        """
        ps = PointStamped()
        ps.header = self._cmd_pose.header
        ps.header.frame_id = 'world'
        ps.point = self._cmd_pose.pose.position
        ros_time_now = rospy.Time.now()
        ps.header.stamp = ros_time_now

        try:
            self._tf_listener.waitForTransform('world',
                                               self.robot_frame_id,
                                               ros_time_now,
                                               rospy.Duration(WAIT_DURATION))
            robot_frame_cmd_pos = self._tf_listener.transformPoint(
                self.robot_frame_id, ps)

            world_frame_cmd_yaw = self._cmd_pose.pose.orientation.z

            robot_xerr = robot_frame_cmd_pos.point.y
            robot_yerr = robot_frame_cmd_pos.point.x
            robot_werr = world_frame_cmd_yaw - ardrone.yaw

            time_now = rospy.Time.now()
            dt = (time_now - self._last_time).to_sec()
            self._xe_int +=  robot_xerr*dt
            self._ye_int += robot_yerr*dt
            self._last_time = time_now

            # Robot velocity in the world
            world_xve = ardrone.twist.twist.linear.x
            world_yve = ardrone.twist.twist.linear.y

            world_xverr = self._cmd_twist.twist.linear.x - world_xve
            world_yverr = self._cmd_twist.twist.linear.y - world_yve

            # Velocity rotation matrix
            wr_theta = ardrone.yaw + np.pi/2
            wr_rot = np.array([[np.cos(wr_theta), np.sin(wr_theta)],
                               [np.sin(wr_theta), -np.cos(wr_theta)]])

            world_verr = np.array([world_xverr, world_yverr], ndmin=2).T
            robot_verr = wr_rot.dot(world_verr)

            xverr = robot_verr[0]
            yverr = robot_verr[1]
            """
            # Convert world velocity to robot's frame
            world_vel = np.array([world_xve, world_yve], ndmin=2).T
            robot_vel = wr_rot.dot(world_vel)
            robot_xve = robot_vel[0]
            robot_yve = robot_vel[1]

            xverr = self._cmd_twist.twist.linear.x - robot_xve
            yverr = self._cmd_twist.twist.linear.y - robot_yve
            """
#            xve = (world_xve*np.cos(ardrone.yaw + np.pi/2) +
#                   world_yve*np.sin(ardrone.yaw + np.pi/2))
#            yve = (world_xve*np.sin(ardrone.yaw + np.pi/2) -
#                   world_yve*np.cos(ardrone.yaw + np.pi/2))

#            debugTwist = TwistStamped()
#            debugTwist.header.stamp = rospy.Time.now()
#            debugTwist.twist.linear.x = xve
#            debugTwist.twist.linear.y = yve
#
#            self.debugPub.publish(debugTwist)

#            yve = ardrone.twist.twist.linear.y*np.sin(ardrone.yaw)
#            print('Yaw: {: .5f}'.format(ardrone.yaw))
#            print('Xve: {: .5f}, Yve: {: .5f}'.format(xve, yve))

            x_cmd = self._clamp(self._kp*robot_xerr + self._ki*self._xe_int +
                                self._kd*xverr - self._kangle*ardrone.pitch)

            y_cmd = -self._clamp(self._kp*robot_yerr + self._ki*self._ye_int +
                                 self._kd*yverr - self._kangle*ardrone.roll)

            w_cmd = self._clamp(self._kpw*robot_werr)

            return x_cmd, y_cmd, w_cmd

        except tf.ExtrapolationException as e:
            rospy.loginfo(e)
            return 0, 0, 0


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
