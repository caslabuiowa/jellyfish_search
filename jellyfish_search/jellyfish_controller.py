import sys

import numpy as np
import rclpy
from rclpy.node import Node

from jellyfish_search_msgs.msg import BernsteinTrajectory, BernsteinTrajectoryArray

from jfs_core.jfs import JellyfishSearch


class TrajectoryGenerator(Node):
    def __init__(self):
        super().__init__('trajectory_generator')

        self.traj_pub = self.create_publisher(BernsteinTrajectory, 'bebot_trajectory', 10)
        self.traj_array_pub = self.create_publisher(BernsteinTrajectoryArray, 'bebot_trajectory_array', 10)

        self.jfs = JellyfishSearch(rng_seed=1, num_workers=10)


def project_goal(x, rsafe, goal):
    x = x.squeeze()
    goal = goal.squeeze()
    u = np.linalg.norm(goal - x)
    scale = rsafe / u

    if scale < 1:
        new_goal = (goal - x)*scale + x
    else:
        new_goal = goal

    return new_goal


def main(args=None):
    rclpy.init(args=args)
    trajectory_generator = TrajectoryGenerator()

    try:
        rclpy.spin(trajectory_generator)
    except KeyboardInterrupt:
        trajectory_generator.get_logger().info('Keyboard interrupt. Quitting.')
    finally:
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        del trajectory_generator.jfs
        trajectory_generator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
