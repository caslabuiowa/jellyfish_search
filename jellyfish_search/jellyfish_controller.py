import sys

import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from geometry_msgs.msg import Point, PoseStamped
from jellyfish_search_msgs.msg import BernsteinTrajectory, BernsteinTrajectoryArray, ObstacleArray
from nav_msgs.msg import Odometry

from BeBOT.polynomial.bernstein import Bernstein
from jfs_core.jfs import JellyfishSearch, _feasibility_check


#TODO
# x Test the JFS with obstacles (probably a good idea to spawn them within the VRX simulator)
# x Make follow up trajectories continuous
# * Decide how to address the perturbed goal w.r.t. the safe sphere
# * Fix vmax and wmax issues by changing tf rather than culling the trajs that violate
# x Adaptive noise on goal proximity
# * v, w to LR thrust controller


class TrajectoryGenerator(Node):
    def __init__(self):
        super().__init__('trajectory_generator')

        # Parameters
        percep_rad_desc = ParameterDescriptor(
            description='Radius at which the vehicle can detect obstacles in meters.')
        rng_seed_desc = ParameterDescriptor(
            description='Seed value to pass to the random number generator. Pass a negative number for no seed value.')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('perception_radius', 10.0, percep_rad_desc),
                ('rng_seed', 1),
                ('num_workers', 10),
                ('num_trajectories', 100),
                ('maximum_speed', 10.0),
                ('maximum_angular_rate', np.pi/4),
                ('obstacle_position_std', 0.0),
                ('obstacle_size_std', 0.3),
                ('goal_position_std', 6.0),
                ('polynomial_degree', 5),
                ('trajectory_frame_id', 'odom'),
                ('vehicle_frame_id', 'base_link'),
                ('trajectory_generation_period', 1.0),
                ('solver_params.n_steps', 300),
                ('solver_params.tf_max', 60.0,),
                ('solver_params.Katt', 1.0,),
                ('solver_params.Krep', 1.0,),
                ('solver_params.rho_0', 1.0,),
                ('solver_params.delta', 0.0),
                ('obstacle_topic', 'obstacles'),
                ('pose_topic', 'odometry'),
                ('goal_topic', 'goal'),
                ('publish_traj_array', False)],
            )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.traj_pub = self.create_publisher(BernsteinTrajectory, 'bebot_trajectory', 10)
        if self.get_parameter('publish_traj_array').value:
            self.traj_array_pub = self.create_publisher(BernsteinTrajectoryArray, 'bebot_trajectory_array', 10)
        self.obs_dist_pub = self.create_publisher(BernsteinTrajectoryArray, 'obstacle_distances', 10)

        self.obs_sub = self.create_subscription(ObstacleArray,
                                                self.get_parameter('obstacle_topic').value,
                                                self.obstacle_cb, 10)
        self.pos_sub = self.create_subscription(Odometry,
                                                self.get_parameter('pose_topic').value,
                                                self.pose_cb, QoSProfile(depth=10,
                                                                         reliability=QoSReliabilityPolicy.BEST_EFFORT))
        self.goal_sub = self.create_subscription(PoseStamped,
                                                 self.get_parameter('goal_topic').value,
                                                 self.goal_cb, 10)

        self.pose = None
        self.goal = None
        # self.previous_trajectory = None
        # self.previous_time = None
        self.x_pred = None

        self.obstacles = np.array([[]], dtype=float)
        self.obstacle_safe_distances = np.array([], dtype=float)

        rng_seed = self.get_parameter('rng_seed').value
        if rng_seed < 0:
            rng_seed = None
        self.jfs = JellyfishSearch(rng_seed=rng_seed,
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
        # try:
        #     to_frame_rel = 'map' #self.get_parameter('vehicle_frame_id').value
        #     from_frame_rel = data.header.frame_id
        #     xfrm = self._tf_buffer.lookup_transform(to_frame_rel,
        #                                             from_frame_rel,
        #                                             rclpy.time.Time())
        # except TransformException as e:
        #     msg = f'Unable to transform goal in frame {from_frame_rel} into vehicle frame {to_frame_rel}:\n{e}'
        #     self.get_logger().warning(msg)
        #     return

        # # Translation into the vehicle's frame
        # goal_x_xfrm = data.pose.position.x - xfrm.transform.translation.x
        # goal_y_xfrm = data.pose.position.y - xfrm.transform.translation.y

        # # Rotation into the vehicle's frame
        # q_goal = R.from_quat([data.pose.orientation.x,
        #                       data.pose.orientation.y,
        #                       data.pose.orientation.z,
        #                       data.pose.orientation.w])
        # q_r = R.from_quat([xfrm.transform.rotation.x,
        #                    xfrm.transform.rotation.y,
        #                    xfrm.transform.rotation.z,
        #                    xfrm.transform.rotation.w])
        # q_goal_xfrm = q_r.inv()*q_goal


        # _, _, theta = q_goal_xfrm.as_rotvec()
        # self.goal = np.array([goal_x_xfrm,
        #                       goal_y_xfrm,
        #                       theta],  # Note, for now theta is unused
        #                      dtype=float)
        self.goal = np.array([data.pose.position.x,
                              data.pose.position.y,
                              0], dtype=float)

    def main_cb(self):
        if self.pose is None:
            self.get_logger().info('Waiting for initial pose.')
        elif self.goal is None:
            self.get_logger().info('Waiting for goal.')
        else:
            self.get_logger().info('=======================')
            #TODO: Perception radius vs safe planning radius. The goal is perturbed a ton which would bring a lot of
            # trajectories outside of the safe sphere resulting in many infeasible trajectories. For now, rsafe is
            # being set to a very large number, effectively negating it
            rsafe = 100

            # Use our initial pose for x0 if we haven't generated a trajectory yet. Otherwise, always use past
            # trajectories to maintain continuity
            if self.x_pred is None:
                x0 = self.pose[:2]
            else:
                x0 = self.x_pred

            perception_rad = self.get_parameter('perception_radius').value
            cur_goal = project_goal(x0, perception_rad, self.goal[:2])
            solver_params = dict(n_steps=self.get_parameter('solver_params.n_steps').value,
                                 tf_max=self.get_parameter('solver_params.tf_max').value,
                                 Katt=self.get_parameter('solver_params.Katt').value,
                                 Krep=self.get_parameter('solver_params.Krep').value,
                                 rho_0=self.get_parameter('solver_params.rho_0').value,
                                 delta=self.get_parameter('solver_params.delta').value)

            obstacles = self.obstacles.copy()
            obstacle_safe_distances = self.obstacle_safe_distances.copy()
            num_trajectories = self.get_parameter('num_trajectories').value
            vmax = self.get_parameter('maximum_speed').value
            wmax = self.get_parameter('maximum_angular_rate').value
            #TODO: Remove obstacle position perturbation entirely maybe? Is there a use case for this?
            obs_pos_std = self.get_parameter('obstacle_position_std').value
            obs_size_std = self.get_parameter('obstacle_size_std').value
            goal_pos_std = self.get_parameter('goal_position_std').value
            degree = self.get_parameter('polynomial_degree').value

            # if self.previous_trajectory is None:
            #     x0 = self.pose[:2]
            # else:
            #     t = self.get_parameter('trajectory_generation_period').value
            #     # t = (self.get_clock().now() - self.previous_time).nanoseconds*1e-9 + dt
            #     if t > self.previous_trajectory.tf:
            #         t = self.previous_trajectory.tf
            #     elif t < self.previous_trajectory.t0:
            #         t = self.previous_trajectory.t0
            #     x0 = self.previous_trajectory(t).squeeze()


            d_goal = np.linalg.norm(cur_goal - x0)
            goal_pos_std *= (1 - np.exp(-d_goal/10))

            self.get_logger().info(f'{x0=}')
            self.get_logger().info(f'{cur_goal=}')
            self.get_logger().info(f'{d_goal=}')
            self.get_logger().info(f'{goal_pos_std=}')
            self.get_logger().info(f'{obstacles=}')
            self.get_logger().info(f'{obstacle_safe_distances=}')

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
                self.get_logger().warning('Best trajectory is infeasible, recomputing.')
                self.main_cb()
            else:
                traj = result[0][0]
                cost = result[0][1]
                self.get_logger().info(f'Best trajectory: {traj}\ncost: {cost}')
                self.get_logger().info(f'Feasibility Check: {_feasibility_check(traj, obstacle_safe_distances, obstacles, vmax, wmax, rsafe)}')

                # Log obstacle distances for the chosen trajectory for debugging
                distances = log_obstacles(traj, obstacles, obstacle_safe_distances)
                obs_dist_msg = BernsteinTrajectoryArray()
                for dist in distances:
                    dist_msg = self.create_bt_msg(dist.cpts, dist.t0, dist.tf)
                    obs_dist_msg.trajectories.append(dist_msg)
                self.obs_dist_pub.publish(obs_dist_msg)

                # Manually compute the distance between the vehicle and obstacles
                for i, dist in enumerate(distances):
                    if np.any((dist.elev(20).cpts.squeeze() - obstacle_safe_distances[i]**2) < 0):
                        print('traj not feasible')
                        self.get_logger().info(f'{(dist.elev(20).cpts.squeeze() - obstacle_safe_distances[i]**2)=}')

                traj_msg = self.create_bt_msg(traj.cpts, traj.t0, traj.tf)
                self.traj_pub.publish(traj_msg)

                t_pred = self.get_parameter('trajectory_generation_period').value
                if t_pred > traj.tf:
                    t_pred = traj.tf
                elif t_pred < traj.t0:
                    t_pred = traj.t0
                # self.previous_trajectory = traj
                # self.previous_time = Time.from_msg(traj_msg.header.stamp)
                self.x_pred = traj(t_pred).squeeze()

            if self.get_parameter('publish_traj_array').value:
                traj_array_msg = BernsteinTrajectoryArray()
                for res in result:
                    traj = res[0]
                    traj_msg = self.create_bt_msg(traj.cpts, traj.t0, traj.tf)
                    traj_array_msg.trajectories.append(traj_msg)
                self.traj_array_pub.publish(traj_array_msg)

            self.get_logger().info('=======================')

    def create_bt_msg(self, cpts, t0, tf):
        bt = BernsteinTrajectory()
        bt.header.frame_id = self.get_parameter('trajectory_frame_id').value
        bt.header.stamp = self.get_clock().now().to_msg()

        bt.t0 = float(t0)
        bt.tf = float(tf)
        try:
            bt.cpts = [Point(x=pt[0], y=pt[1]) for pt in cpts.T]
        except IndexError:
            bt.cpts = [Point(x=pt[0]) for pt in cpts.T]

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


def log_obstacles(traj, obstacles, obstacle_safe_distances):
    ndim = traj.dim
    n = traj.deg
    t0 = traj.t0
    tf = traj.tf

    distances = []
    for i, obs in enumerate(obstacles):
        cpts = np.array([[j]*(n+1) for j in obs[:ndim]], dtype=float)
        c_obs = Bernstein(cpts, t0, tf)

        dist = (traj - c_obs).normSquare()
        distances.append(dist)

    return distances


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
