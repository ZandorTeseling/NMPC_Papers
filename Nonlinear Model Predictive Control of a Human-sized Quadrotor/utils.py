#
# Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
# Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
# Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
# Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def QuattoRPY(q):
    """
    Params:
    In:
        q: N x 4
    Out
        rpy: N x 3
    """


    if q.ndim == 1:
        q = q.reshape((1, 4))
    N = q.shape[0]
    rpy = np.zeros((N, 3))

    for i in range(N):
        #roll (x-axis rotation)

        sinr_cosp = 2 * (q[i, 0] * q[i, 1] + q[i, 2] * q[i, 3])

        cosr_cosp = 1 - 2 * (q[i, 1] * q[i, 1] + q[i, 2] * q[i, 2])
        rpy[i, 0] = np.arctan2(sinr_cosp, cosr_cosp)

        #pitch (y-axis rotation)
        sinp = 2 * (q[i, 0] * q[i, 2] - q[i, 3] * q[i, 1])
        if (abs(sinp) >= 1):
            rpy[i, 1] = np.sign(sinp) * np.pi # use 90 degrees if out of range
        else:
            rpy[i, 1] = np.arcsin(sinp)

        # yaw (z-axis rotation)
        siny_cosp = 2 * (q[i, 0] * q[i, 3] + q[i, 1] * q[i, 2])
        cosy_cosp = 1 - 2 * (q[i, 2] * q[i, 2] + q[i, 3] * q[i, 3])
        rpy[i, 2] = np.arctan2(siny_cosp, cosy_cosp)

    return rpy

#Convertion for the ZYX sequence of rotation. RPY--from right to left--
# xa vector in basis {a}
# xb vector in basis {b}
# Rab = dcm of basis {b} w.r.t basis{a}
# xa = R_z(\psi)R_y(\theta)R_X(\phi) xb
#
def YPRtoQuat(rpy):
    """
    Params:
    In:
        rpy: N x 3
    Out
        q: N x 4
    """

    if rpy.ndim == 1:
        rpy = rpy.reshape((1, 3))
    N = rpy.shape[0]
    q = np.zeros((N, 4))

    for i in range(N):
        cy = np.cos(rpy[i, 2] * 0.5)
        sy = np.sin(rpy[i, 2] * 0.5)
        cp = np.cos(rpy[i, 1] * 0.5)
        sp = np.sin(rpy[i, 1] * 0.5)
        cr = np.cos(rpy[i, 0] * 0.5)
        sr = np.sin(rpy[i, 0] * 0.5)

        #Quaternion
        q[i, 0] = cy * cp * cr + sy * sp * sr
        q[i, 1] = cy * cp * sr - sy * sp * cr
        q[i, 2] = sy * cp * sr + cy * sp * cr
        q[i, 3] = sy * cp * cr - cy * sp * sr

    return q

def QuattoR(q):
    """
    Params:
    In:
        q: 1 x 4
    Out
        R: 3 x 3
    """
    R = np.zeros((3, 3))
    if q.shape[0] != 1:
        print("Error:", q.shape[0])
        return R
    q0 = q[0, 0]
    q1 = q[0, 1]
    q2 = q[0, 2]
    q3 = q[0, 3]

    R[0, 0] = 1 - 2*(q2**2 + q3**2)
    R[0, 1] = 2*(q1*q2 - q3*q0)
    R[0, 2] = 2*(q1*q3 + q2*q0)

    R[1, 0] = 2*(q1*q2 + q3*q0)
    R[1, 1] = 1 - 2*(q1**2 + q3**2)
    R[1, 2] = 2*(q2*q3 - q1*q0)

    R[2, 0] = 2*(q1*q3 - q2*q0)
    R[2, 1] = 2*(q2*q3 + q1*q0)
    R[2, 2] = 1 - 2*(q1**2 + q2**2)

    return R

def plot_quad(h, U_ss, U_del, U, X_true, X_est=None, Y_measured=None, latexify=True):
    """
    Params:
        h: time step
        u_max: maximum absolute value of u
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X_true: arrray with shape (N_sim, nx)
        X_est: arrray with shape (N_sim-N_mhe, nx)
        Y_measured: array with shape (N_sim, ny)
        latexify: latex style plots
    """

    # latexify plot
    if latexify:
        params = {'backend': 'ps',
                'text.latex.preamble': [r"\usepackage{gensymb} \usepackage{amsmath}"],
                'axes.labelsize': 15,
                'axes.titlesize': 15,
                'legend.fontsize': 15,
                'xtick.labelsize': 15,
                'ytick.labelsize': 15,
                'text.usetex': True,
                'font.family': 'serif'
        }

        matplotlib.rcParams.update(params)

    WITH_ESTIMATION = X_est is not None and Y_measured is not None

    N_sim = X_true.shape[0]
    nx = X_true.shape[1]

    Tf = N_sim*h
    t = np.linspace(0.0, Tf, N_sim)

    control_labels = ['$w_1$',
                      '$w_2$',
                      '$w_3$',
                      '$w_4$']

    if WITH_ESTIMATION:
        N_mhe = N_sim - X_est.shape[0]
        t_mhe = np.linspace(N_mhe, Tf, N_sim)

    ##Control Figure
    ########################################################
    plt.figure(1)
    plt.subplot(1, 1, 1)
    plt.step(t[:U.shape[0]], U,'o-')
    plt.title('closed-loop control inputs')
    plt.ylabel('$u$')
    plt.xlabel('$t$')
    plt.hlines(U_ss + U_del, t[0], t[-2], linestyles='dashed', alpha=0.7)
    plt.hlines(U_ss - U_del, t[0], t[-2], linestyles='dashed', alpha=0.7 )
    plt.ylim((U_ss - 1.2*U_del,U_ss + 1.2*U_del))
    plt.grid()
    plt.legend(control_labels)

    quat_labels = ['$q_0$',
                   '$q_1$',
                   '$q_2$',
                   '$q_3$']

    ##Quaternion Figure
    ########################################################
    plt.figure(2)
    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plt.plot(t, X_true[:, i], label='true')
        plt.ylabel(quat_labels[i])
        plt.xlabel('$t$')
        plt.grid()
        plt.figure(2).suptitle("Quaternions")
        plt.ylim([1, -1])
        plt.legend(loc=1)


    ##Euler and Omega Figure
    ########################################################
    #Remapping x = [q0,q1,q2,q3,omega_x,omega_y,omega_z] -> [phi,theta,psi,omega_x,omega_y,omega_z]
    X_remapped = np.zeros((N_sim,nx-1))
    X_remapped[:, 0:3] = QuattoRPY(X_true[:,0:4])
    X_remapped[:, 3:nx-1] = X_true[:, 4:nx]
    # X_remapped[:, 4] = X_true[:, 5]
    # X_remapped[:, 5] = X_true[:, 6]

    states_lables = ['$\phi_{roll}$',
                     '$\\theta_{pitch}$',
                     '$\psi_{yaw}$',
                     '$\omega_x$',
                     '$\omega_y$',
                     '$\omega_z$']
    plt.figure(3)
    plt.figure(3).suptitle("RPY[rads] and Angular Velocity[rad/sec]")

    for i in range(nx -1):
        plt.subplot(nx - 1, 1, i + 1)
        plt.plot(t, X_remapped[:,i], label='true')

        if WITH_ESTIMATION:
            plt.plot(t_mhe, X_est[:, i], label='estimated')
            plt.plot(t, Y_measured[:, i], 'x', label='measured')

        plt.ylabel(states_lables[i])
        plt.xlabel('$t$')
        plt.grid()
        plt.legend(loc=1)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

    # avoid plotting when running on Travis
    if os.environ.get('ACADOS_ON_TRAVIS') is None:
        plt.show()
