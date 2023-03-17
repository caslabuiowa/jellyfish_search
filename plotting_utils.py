#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 21:24:44 2022

@author: magicbycalvin
"""

import matplotlib.pyplot as plt
import numpy as np

from optimization.AngularRate import angularRate
from optimization.ObstacleAvoidance import obstacleAvoidance
from optimization.Speed import speed
from optimization.TemporalSeparation import temporalSeparation
from polynomial.bernstein import Bernstein

def setRCParams():
    """Set matplotlib's RC params for CAS Lab work.

    Run this to make sure that the matplotlib plots have the correct font type
    for an IEEE publication. Also sets font sizes and line widths for easier
    viewing.

    Returns
    -------
    None.

    """
    plt.rcParams.update({
                'font.size': 40,
                'figure.titlesize': 40,
                'pdf.fonttype': 42,
                'ps.fonttype': 42,
                'xtick.labelsize': 40,
                'ytick.labelsize': 40,
                'lines.linewidth': 4,
                'lines.markersize': 18,
                'figure.figsize': [13.333, 10]
                })



def resetRCParams():
    """Reset matplotlib's RC parameters to default

    Returns
    -------
    None.

    """
    plt.rcParams.update(plt.rcParamsDefault)


def save_all_figures(fig_names=None):
    for i, n in enumerate(plt.get_fignums()):
        fig = plt.figure(n)
        fig.tight_layout()

        if fig_names is None:
            fig_name = f'figure_{i}'
        else:
            fig_name = fig_names[i]

        fig.savefig(fig_name+'.svg', dpi=300, format='svg')


#TODO
def plotConstraints(trajs, tf, params):
    XLIM = [0-0.1*tf, tf+0.1*tf]

    # Speed constraints
    speedFig, speedAx = plt.subplots()
    legS = ('$v_{1}^2 (t)$', '$v_{2}^2 (t)$', '$v_{3}^2 (t)$')
    for i, traj in enumerate(trajs):
        xdot = traj.diff().x
        ydot = traj.diff().y
        speed = xdot*xdot + ydot*ydot
        speed.plot(speedAx, showCpts=False, label=legS[i])
    speedAx.plot(XLIM, [params.vMax**2]*2, '--', label=r'$v^2_{max}$')
    speedAx.set_xlim(XLIM)
    speedAx.legend(fontsize=32)
    speedAx.set_xlabel('Time (s)')
    speedAx.set_ylabel(r'Squared Speed $\left( \frac{m}{s} \right)^2$')
    speedAx.set_title('Speed Constraints')

    # Inter-vehicle safety
    legD = ('$||p_1 (t) - p_2 (t)||^2$', '$||p_1 (t) - p_3 (t)||^2$', '$||p_2 (t) - p_3 (t)||^2$')
    collFig, collAx = plt.subplots()
    dist = temporalSeparation(trajs)
    dist_cpnt = np.reshape(dist, (params.N_V, int(len(dist)/params.N_V)))
    dist = []
    for i in range(params.N_V):
        dist.append(Bernstein(dist_cpnt[i,:], tf=tf))
    for i, traj in enumerate(trajs):
        dist[i].plot(collAx, showCpts=False, label=legD[i])
    collAx.plot(XLIM, [params.safeDist**2]*2, '--', label=r'$d_{safe}^2$')
    collAx.set_xlim(XLIM)
    collAx.legend(fontsize=32)
    collAx.set_xlabel('Time (s)')
    collAx.set_ylabel('$||p_i (t) - p_j (t)||^2 \quad [m^2]$')
    collAx.set_title('Inter-vehicle safety Constraints')
