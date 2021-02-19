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

from acados_template import AcadosModel
from casadi import *
import numpy as np
def export_pend_ode_model():

    model_name = 'pend_ode'

    # set up states & controls
    q1  = SX.sym('q1')
    q2  = SX.sym('q2')
    dq1 = SX.sym('dq1')
    dq2 = SX.sym('dq2')
    b1  = SX.sym('b1')
    b2  = SX.sym('b2')
    x = vertcat(q1, q2, dq1, dq2, b1, b2)

    # controls
    tau1 = SX.sym('tau1')
    tau = vertcat(tau1, 0)

    u = vertcat(tau1)
    
    # xdot
    q1_dot  = SX.sym('q1_dot')
    q2_dot  = SX.sym('q2_dot')
    dq1_dot = SX.sym('dq1_dot')
    dq2_dot = SX.sym('dq2_dot')
    b1_dot  = SX.sym('b1_dot')
    b2_dot  = SX.sym('b2_dot')

    xdot = vertcat(q1_dot, q2_dot, dq1_dot, dq2_dot, b1_dot, b2_dot)

    # algebraic variables
    # z = None

    # parameters
    # set up parameters
    m1  = SX.sym('m1')   # mass link 1
    m2  = SX.sym('m2')   # mass link 2
    l1  = SX.sym('l1')   # length link 1
    l2  = SX.sym('l2')   # length link 2
    lc1 = SX.sym('lc1')  # cm link 1
    lc2 = SX.sym('lc2')  # cm link 2
    g   = SX.sym('g')

    I1 = (1 / 3)* m1 * l1 * l1
    I2 = (1 / 3)* m2 * l2 * l2
    p = vertcat(m1, m2, l1, l2, lc1, lc2, g)

    # system dynamics.
    theta1 = m1*lc1*lc1 + m2*l1*l1 + I1
    theta2 = m2*lc2*lc2 + I2
    theta3 = m2*l1*lc2
    theta4 = m1*lc1 + m2*l1
    theta5 = m2*lc2

    # In matrix form, D(q) -> Inertia Matrix, C(dq,q) ->Coriolis Matrix, F(dq) -> Dissipation matrix
    # D(q) ddq + C(q,dq) dq + F(dq) + g(q) = tau
    D = SX.zeros(2, 2)
    C = SX.zeros(2, 2)
    F = SX.zeros(2, 1)
    gVec = SX.zeros(2, 1)

    D[0, 0] = theta1 + theta2 + 2*theta3*cos(q2)
    D[1, 0] = theta2 + theta3*cos(q2)
    D[0, 1] = D[1, 0]
    D[1, 1] = theta2

    C[0, 0] = -theta3*sin(q2)*dq2
    C[1, 0] = -theta3*sin(q2)*dq2 - theta3*sin(q2)*dq1
    C[0, 1] = theta3*sin(q2)*dq1
    C[1, 1] = 0

    F[0, 0] = b1*dq1
    F[1, 0] = b2*dq2

    gVec[0, 0] = theta4*g*cos(q1) + theta5*g*cos(q1+q2)
    gVec[1, 0] = theta5*g*cos(q1+q2)
    # dynamics
    f_expl = vertcat(
                    dq1,
                    dq2,
                    mtimes(inv(D), tau - mtimes(C, vertcat(dq1, dq2)) - F - gVec),
                    0,
                    0
                     )

    f_impl = xdot - f_expl
    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = []
    model.p = p
    model.name = model_name

    return model

