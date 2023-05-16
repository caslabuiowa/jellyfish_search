import sys

import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import Point, PoseStamped
from jellyfish_search_msgs.msg import BernsteinTrajectory, BernsteinTrajectoryArray, ObstacleArray
from nav_msgs.msg import Odometry

from jfs_core.jfs import JellyfishSearch


class TrajectoryGenerator(Node):
    def __init__(self):
        super().__init__('trajectory_generator')

        # Parameters
        percep_rad_description = ParameterDescriptor(
            description='Radius at which the vehicle can detect obstacles in meters.')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('perception_radius', 10, percep_rad_description),
                ('rng_seed', 1),
                ('num_workers', 10),
                ('num_trajectories', 100),
                ('maximum_speed', 10),
                ('maximum_angular_rate', np.pi/4),
                ('obstacle_position_std', 0.0),
                ('obstacle_size_std', 0.3),
                ('goal_position_std', 6.0),
                ('polynomial_degree', 5),
                ('trajectory_frame_id', 'odom'),
                ('trajectory_generation_period', 1.0),
                ('solver_params.n_steps', 300),
                ('solver_params.tf_max', 60.0,),
                ('solver_params.Katt', 1.0,),
                ('solver_params.Krep', 1.0,),
                ('solver_params.rho_0', 1.0,),
                ('solver_params.delta', 0.0)],
            )

        self.traj_pub = self.create_publisher(BernsteinTrajectory, 'bebot_trajectory', 10)
        self.traj_array_pub = self.create_publisher(BernsteinTrajectoryArray, 'bebot_trajectory_array', 10)

        self.obs_sub = self.create_subscription(ObstacleArray, 'obstacles', self.obstacle_cb, 10)
        self.pos_sub = self.create_subscription(Odometry, 'odometry/local', self.pose_cb, 10)
        self.goal_sub = self.create_subscription(PoseStamped, 'goal', self.goal_cb, 10)

        self.pose = None
        self.goal = None
        self.previous_trajectory = None

        self.obstacles = np.array([[]], dtype=float)
        self.obstacle_safe_distances = np.array([], dtype=float)

        self.jfs = JellyfishSearch(rng_seed=self.get_parameter('rng_seed').value,
                                   num_workers=self.get_parameter('num_workers').value)
        self.main_timer = self.create_timer(self.get_parameter('trajectory_generation_period').value, self.main_cb)

    def obstacle_cb(self, data: ObstacleArray):
        obstacles_tmp = []
        obstacles_rad_tmp = []
        for obs in data.obstacles:
            obstacles_tmp.append([obs.position.x, obs.position.y])
            obstacles_rad_tmp.append(obs.radius)

        self.obstacles = np.array(obstacles_tmp)
        self.obstacle_safe_distances = np.array(obstacles_rad_tmp)

    def pose_cb(self, data: Odometry):
        r = R.from_quat([data.pose.pose.orientation.x,
                         data.pose.pose.orientation.y,
                         data.pose.pose.orientation.z,
                         data.pose.pose.orientation.w])
        _, _, theta = r.as_rotvec()
        self.pose = np.array([data.pose.pose.position.x,
                              data.pose.pose.position.y,
                              theta],
                             dtype=float)

    def goal_cb(self, data: PoseStamped):
        r = R.from_quat([data.pose.orientation.x,
                         data.pose.orientation.y,
                         data.pose.orientation.z,
                         data.pose.orientation.w])
        _, _, theta = r.as_rotvec()
        self.goal = np.array([data.pose.position.x,
                              data.pose.position.y,
                              theta],  # Note, for now theta is unused
                             dtype=float)

    def main_cb(self):
        if self.pose is None:
            self.get_logger().info('Waiting for initial pose.')
        elif self.goal is None:
            self.get_logger().info('Waiting for goal.')
        else:
            #TODO: Perception radius vs safe planning radius. The goal is perturbed a ton which would bring a lot of
            # trajectories outside of the safe sphere resulting in many infeasible trajectories. For now, rsafe is
            # being set to a very large number, effectively negating it
            rsafe = 100
            perception_rad = self.get_parameter('perception_radius').value
            cur_goal = project_goal(self.pose[:2], perception_rad, self.goal[:2])
            solver_params = dict(n_steps=self.get_parameter('solver_params.n_steps').value,
                                 tf_max=self.get_parameter('solver_params.tf_max').value,
                                 Katt=self.get_parameter('solver_params.Katt').value,
                                 Krep=self.get_parameter('solver_params.Krep').value,
                                 rho_0=self.get_parameter('solver_params.rho_0').value,
                                 delta=self.get_parameter('solver_params.delta').value)
            x0 = self.pose[:2]
            obstacles = self.obstacles
            obstacle_safe_distances = self.obstacle_safe_distances
            num_trajectories = self.get_parameter('num_trajectories').value
            vmax = self.get_parameter('maximum_speed').value
            wmax = self.get_parameter('maximum_angular_rate').value
            #TODO: Remove obstacle position perturbation entirely maybe? Is there a use case for this?
            obs_pos_std = self.get_parameter('obstacle_position_std').value
            obs_size_std = self.get_parameter('obstacle_size_std').value
            goal_pos_std = self.get_parameter('goal_position_std').value
            degree = self.get_parameter('polynomial_degree').value

            result = self.jfs.generate_trajectories(x0,
                                                    cur_goal,
                                                    obstacles,
                                                    obstacle_safe_distances,
                                                    num_trajectories,
                                                    vmax,
                                                    wmax,
                                                    rsafe,
                                                    obs_pos_std,
                                                    obs_size_std,
                                                    goal_pos_std,
                                                    degree,
                                                    solver_params)

            if result[0][1] == np.inf:
                self.get_logger().warning('Best trajectory is infeasible')
            else:
                traj = result[0][0]
                cost = result[0][1]
                self.get_logger().info(f'Best trajectory: {traj}\ncost: {cost}')

                traj_msg = self.create_bt_msg(traj.cpts, traj.t0, traj.tf)
                self.traj_pub.publish(traj_msg)

    def create_bt_msg(self, cpts, t0, tf):
        bt = BernsteinTrajectory()
        bt.header.frame_id = self.get_parameter('trajectory_frame_id').value
        bt.header.stamp = self.get_clock().now().to_msg()

        bt.t0 = float(t0)
        bt.tf = float(tf)
        bt.cpts = [Point(x=pt[0], y=pt[1]) for pt in cpts.T]

        return bt


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
