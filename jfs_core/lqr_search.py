#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:31:49 2022

@author: magicbycalvin
"""

import logging

from control import lqr
import matplotlib.pyplot as plt
from numba import cfunc, carray
from numbalsoda import lsoda_sig, lsoda
import numpy as np
from numpy.random import default_rng

from polynomial.bernstein import Bernstein
# from stoch_opt.constraint_functions import CollisionAvoidance, MaximumSpeed, MaximumAngularRate, SafeSphere
# from stoch_opt.cost_functions import SumOfDistance
# from stoch_opt.utils import state2cpts


LOG_LEVEL = logging.DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
if len(logger.handlers) < 1:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LOG_LEVEL)
    logger.addHandler(stream_handler)


def generate_lqr_trajectories(x0, goal, Q_std, R_std, x0_std, tf, n=5, rng=None):
    if rng is None:
        rng = default_rng()

    logger.debug('Initializing LQR problem')
    A, B, Q, R = initialize_lqr_problem(Q_std, R_std, rng)
    logger.debug('Perturbing initial state')
    x0_perturbed = perturb_initial_state(x0, goal, x0_std, rng)

    logger.debug((f'{A=}\n'
                  f'{B=}\n'
                  f'{Q=}\n'
                  f'{R=}\n'
                  f'{x0=}\n'
                  f'{x0_perturbed=}'))

    traj = solve_lqr_problem(A, B, Q, R, x0_perturbed, n, tf, goal)

    return traj


def initialize_lqr_problem(Q_std, R_std, rng):
    A = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
    B = np.array([[0, 0],
                  [0, 0],
                  [0, 0],
                  [1, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 1]], dtype=float)

    # Q = rng.normal(0, Q_std, (8, 8))
    # R = rng.normal(0, R_std, (2, 2))
    # Q = (0.5*Q.T@Q).round(6)
    # R = (0.5*R.T@R).round(6)
    Q = np.eye(8)
    R = np.eye(2)

    return A, B, Q, R


def perturb_initial_state(x0, goal, std, rng):
    x0_perturbed = x0.copy()
    # Subtract goal because the LQR controller tries to bring all states back to zero
    x0_perturbed += rng.normal([-goal[0], 0, 0, 0,
                                -goal[1], 0, 0, 0],
                               std)

    return x0_perturbed


def solve_lqr_problem(A, B, Q, R, x0, n, tf, goal):
    K, S, E = lqr(A, B, Q, R)

    usol, success = lsoda(fn.address, x0, np.linspace(0, tf, n+1), data=(A - B@K))

    # cpts = np.concatenate([[usol.T[:, 0] - usol.T[0, 0] + initial_position[0]],
    #                        [usol.T[:, 4] - usol.T[0, 4] + initial_position[1]]], axis=0)
    logger.debug(f'{usol=}')
    cpts = np.concatenate([[usol[:, 0]],
                           [usol[:, 4]]], axis=0)
    cpts += goal[:, np.newaxis]
    logger.debug(f'{cpts=}')
    traj = Bernstein(cpts, tf=tf)

    return traj


@cfunc(lsoda_sig)
def fn(t, u, du, p):
    u_ = carray(u, (8,))
    p_ = carray(p, (8, 8))
    tmp = p_@u_
    for i in range(8):
        du[i] = tmp[i]


# def jellyfish_search(x0, tf, goal, rng, obstacles=[], safe_dist=1, vmax=1, wmax=np.pi/4, rsafe=3, n=7, t_max=1.0):
#     rospy.logdebug('Starting jellyfish search')
#     trajs = []
#     best_cost = np.inf

#     w = Worker(n, t0=0, tf=tf, rsafe=rsafe, obstacles=obstacles)

#     solutions = []
#     count = 0
#     tstart = rospy.get_time()
#     while rospy.get_time() - tstart < t_max:
#         # x0 = np.array([rng.normal(initial_guess.cpts[0, 0] - goal[0], 0.1), x0dot[0], x0ddot[0], x0dddot[0],
#         #                rng.normal(initial_guess.cpts[1, 0] - goal[1], 0.1), x0dot[1], x0ddot[1], x0dddot[1]],
#         #               dtype=float)
#         # x0 = np.array([rng.normal(x0[0] - goal[0], 0.1), x0[1], x0[2], x0[3],
#         #                rng.normal(x0[4] - goal[1], 0.1), x0dot[1], 0, 0], dtype=float)
#         x0_perturbed = x0.copy()
#         x0_perturbed[:4] -= rng.normal(goal[0], 1.0)
#         x0_perturbed[4:] -= rng.normal(goal[1], 1.0)

#         rospy.loginfo(f'{x0_perturbed=}')

#         Q = rng.normal(0, 10, (8, 8))
#         R = rng.normal(0, 300, (2, 2))

#         cur_cost, cur_traj = w.do_work(x0_perturbed, Q, R, x0[(0, 4),])
#         # cur_traj.cpts[0, :] += initial_guess.cpts[0, 0]
#         # cur_traj.cpts[1, :] += initial_guess.cpts[1, 0]
#         solutions.append((cur_cost, count, cur_traj))
#         # Count is used for situations where all the costs are the same, otherwise min() will try checking the
#         # trajectories against each other and it will throw an error
#         count += 1

#     rospy.logdebug(f'Number of trajectories: {len(solutions)}')

#     # The unused middle value is for avoiding min() checking trajectories against eachother as mentioned above
#     best_cost, _, best_traj = min(solutions)
#     rospy.loginfo(f'Best cost: {best_cost}')

#     return best_traj, best_cost, solutions


# def is_feasible(traj, obstacles, safe_dist, vmax, wmax, rsafe):
#     rospy.logdebug('Inside is_feasible')
#     constraints = [
#         CollisionAvoidance(safe_dist, obstacles, elev=1000),
#         MaximumSpeed(vmax),
#         MaximumAngularRate(wmax),
#         SafeSphere(traj.cpts[:, 0].squeeze(), rsafe)
#     ]

#     rospy.logdebug('Starting feasibility check')
#     for i, cons in enumerate(constraints):
#         rospy.logdebug(f'Constraint {i}')
#         if not cons.call([traj]):
#             return False

#     return True


# def cost_fn(traj, goal):
#     sod = SumOfDistance(goal)

#     return sod.call([traj])


# def project_goal(x, rsafe, goal):
#     x = x.squeeze()
#     goal = goal.squeeze()
#     u = np.linalg.norm(goal - x)
#     scale = rsafe / u

#     if scale < 1:
#         new_goal = (goal - x)*scale + x
#     else:
#         new_goal = goal

#     return new_goal


# def initial_guess(initial_position, initial_velocity, initial_acceleration, t0, tf, n, ndim, goal):
#     cpts_guess = np.empty((ndim, n+1))
#     cpts_guess[:, :3] = state2cpts(initial_position, initial_velocity, initial_acceleration, t0, tf, n)
#     cpts_guess[0, 3:] = np.linspace(cpts_guess[0, 2], goal[0], num=n-1)[1:]
#     cpts_guess[1, 3:] = np.linspace(cpts_guess[1, 2], goal[1], num=n-1)[1:]
#     traj_guess = Bernstein(cpts_guess, t0, tf)

#     return traj_guess


# class Worker:
#     def __init__(self, n, t0=0.0, tf=1.0, rsafe=1.0, obstacles=[], rng_seed=None):
#         self.rng = np.random.default_rng(rng_seed)

#         self.n = n
#         self.t0 = t0
#         self.tf = tf
#         self.rsafe = rsafe
#         self.obstacles = obstacles

#         self.A = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 1, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 1, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 1, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 1, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 1],
#                            [0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
#         self.B = np.array([[0, 0],
#                            [0, 0],
#                            [0, 0],
#                            [1, 0],
#                            [0, 0],
#                            [0, 0],
#                            [0, 0],
#                            [0, 1]], dtype=float)

#     def do_work(self, x0, Q, R, initial_position):
#         rospy.logdebug('In worker')
#         Q = (0.5*Q.T@Q).round(6)
#         R = (0.5*R.T@R).round(6)

#         K, S, E = lqr(self.A, self.B, Q, R)

#         rospy.logdebug('Starting IVP solver')
#         usol, success = lsoda(fn.address, x0, np.linspace(0, tf, self.n+1), data=(self.A - self.B@K))

#         rospy.logdebug('Creating BPs')
#         cpts = np.concatenate([[usol.T[:, 0] - usol.T[0, 0] + initial_position[0]],
#                                [usol.T[:, 4] - usol.T[0, 4] + initial_position[1]]], axis=0)
#         traj = Bernstein(cpts, t0=self.t0, tf=self.tf)

#         rospy.logdebug('Checking feasibility')
#         if is_feasible(traj, self.obstacles, safe_dist, vmax, wmax, self.rsafe):
#             rospy.logdebug('Assigning cost')
#             cost = cost_fn(traj, goal)
#             return cost, traj
#         else:
#             return np.inf, traj


# class JellyfishController:
#     def __init__(self, n, ndim, tf, obstacles, safe_dist, vmax, wmax, safe_planning_radius, t_max, rng_seed=None):
#         rospy.init_node('jellyfish_node')
#         self.traj_pub = rospy.Publisher('trajectory', BernsteinTrajectory, queue_size=10)
#         self.traj_array_pub = rospy.Publisher('jfs_guess', BernsteinTrajectoryArray, queue_size=10)

#         self.n = n
#         self.ndim = ndim
#         self.tf = tf
#         self.obstacles = obstacles
#         self.safe_dist = safe_dist
#         self.vmax = vmax
#         self.wmax = wmax
#         self.safe_planning_radius = safe_planning_radius
#         self.t_max = t_max
#         self.rng_seed = rng_seed
#         self.rng = default_rng(rng_seed)

#         self.rate = rospy.Rate(1)

#         self.A = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 1, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 1, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 1, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 1, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 1],
#                            [0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
#         self.B = np.array([[0, 0],
#                            [0, 0],
#                            [0, 0],
#                            [1, 0],
#                            [0, 0],
#                            [0, 0],
#                            [0, 0],
#                            [0, 1]], dtype=float)

#     def start(self, initial_position, initial_velocity, initial_acceleration, goal, dt):
#         t0 = 0
#         tf = 10
#         while not rospy.is_shutdown():
#             cur_goal = project_goal(initial_position, self.safe_planning_radius, goal)
#             rospy.loginfo(f'Current goal: {cur_goal}')

#             # traj_guess = initial_guess(initial_position, initial_velocity, initial_acceleration, 0, self.tf, self.n,
#             #                            self.ndim, cur_goal)
#             # rospy.loginfo(f'Initial position: {traj_guess.cpts[:, 0]}')

#             x0 = np.array([initial_position[0],
#                            initial_velocity[0],
#                            initial_acceleration[0],
#                            0,
#                            initial_position[1],
#                            initial_velocity[1],
#                            initial_acceleration[1],
#                            0], dtype=float)
#             best_traj, best_cost, trajs = jellyfish_search(x0, self.tf, cur_goal, self.rng,
#                                                            obstacles=self.obstacles,
#                                                            safe_dist=self.safe_dist,
#                                                            vmax=self.vmax,
#                                                            wmax=self.wmax,
#                                                            rsafe=self.safe_planning_radius,
#                                                            n=self.n,
#                                                            t_max=self.t_max)

#             initial_position = best_traj(t0 + dt).squeeze()
#             initial_velocity = best_traj.diff()(t0 + dt).squeeze()
#             initial_acceleration = best_traj.diff().diff()(t0 + dt).squeeze()

#             msg = BernsteinTrajectory()
#             cpts = []
#             for pt in best_traj.cpts.T:
#                 tmp = Point()
#                 tmp.x = pt[0]
#                 tmp.y = pt[1]
#                 cpts.append(tmp)
#             msg.cpts = cpts
#             msg.t0 = t0
#             msg.tf = tf
#             msg.header.stamp = rospy.get_rostime()
#             msg.header.frame_id = 'world'
#             self.traj_pub.publish(msg)

#             # Publish traj here
#             # Vehicle should monitor time so that it follows the new traj asap

#             costs, _, _ = zip(*trajs)
#             traj_idxs = np.argsort(costs)
#             traj_list = []
#             for i in traj_idxs[:10]:
#                 traj = trajs[i][-1]
#                 bern_traj_msg = BernsteinTrajectory()
#                 cpts = []
#                 for pt in traj.cpts.T:
#                     tmp = Point()
#                     tmp.x = pt[0]
#                     tmp.y = pt[1]
#                     cpts.append(tmp)
#                 bern_traj_msg.cpts = cpts
#                 bern_traj_msg.t0 = t0
#                 bern_traj_msg.tf = tf
#                 # bern_traj_msg.header.stamp = rospy.get_rostime()
#                 bern_traj_msg.header.frame_id = 'world'
#                 traj_list.append(bern_traj_msg)

#             traj_array_msg = BernsteinTrajectoryArray()
#             traj_array_msg.trajectories = traj_list
#             self.traj_array_pub.publish(traj_array_msg)

#             self.rate.sleep()
#             # Need to skip rate.sleep if the search doesn't find a solution in time (well, does it matter since the
#             # jf search always lasts a second?)


if __name__ == '__main__':
    seed = 3

    n = 20
    t0 = 0
    tf = 10
    dt = 1
    ndim = 2
    t_max = 0.95
    Q_std = 1#10
    R_std = 1#300
    x0_std = 0

    safe_planning_radius = 10
    safe_dist = 2
    vmax = 3
    wmax = np.pi/4

    obstacles = [np.array([8, 0], dtype=float),  # Obstacle positions (m)
                 np.array([20, 2], dtype=float),
                 np.array([60, 1], dtype=float),
                 np.array([40, 2], dtype=float),
                 np.array([50, -3], dtype=float),
                 np.array([80, -3], dtype=float),
                 np.array([30, -1], dtype=float)]

    goal = np.array([100, 0], dtype=float)
    initial_position = np.array([0, 0], dtype=float)
    initial_velocity = np.array([0, 1], dtype=float)
    initial_acceleration = np.array([0, 0], dtype=float)
    x0 = np.concatenate([[initial_position],
                         [initial_velocity],
                         [initial_acceleration],
                         [[0, 0]]], axis=0).T.reshape(-1)

    # logger.debug('debug')
    # logger.info('info')
    # logger.warning('warning')
    # logger.error('error')
    # logger.critical('critical')

    # jfc = JellyfishController(n, ndim, tf, obstacles, safe_dist, vmax, wmax, safe_planning_radius, t_max)
    # jfc.start(initial_position, initial_velocity, initial_acceleration, goal, dt)
    traj = generate_lqr_trajectories(x0, goal, Q_std, R_std, x0_std, tf, n=n)
    plt.close('all')
    traj.plot()

    ### For debugging only
    # rng = default_rng(seed)
    # A = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
    #               [0, 0, 1, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 1, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 1, 0, 0],
    #               [0, 0, 0, 0, 0, 0, 1, 0],
    #               [0, 0, 0, 0, 0, 0, 0, 1],
    #               [0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
    # B = np.array([[0, 0],
    #               [0, 0],
    #               [0, 0],
    #               [1, 0],
    #               [0, 0],
    #               [0, 0],
    #               [0, 0],
    #               [0, 1]], dtype=float)

    # Q = rng.normal(0, Q_std, (8, 8))
    # R = rng.normal(0, R_std, (2, 2))
