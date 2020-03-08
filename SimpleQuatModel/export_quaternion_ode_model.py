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

def export_quaternion_ode_model():

    model_name = 'quaternion_ode'

    # set up states & controls
    q0 = SX.sym('q0')
    q1 = SX.sym('q1')
    q2 = SX.sym('q2')
    q3 = SX.sym('q3')
    omegax = SX.sym('omegax')
    omegay = SX.sym('omegay')
    omegaz = SX.sym('omegaz')

    x = vertcat(q0, q1, q2, q3, omegax, omegay, omegaz)

    # controls
    omegax_cont = SX.sym('omegax_cont')
    omegay_cont = SX.sym('omegay_cont')
    omegaz_cont = SX.sym('omegaz_cont')

    u = vertcat(omegax_cont, omegay_cont, omegaz_cont)
    
    # xdot
    q0_dot = SX.sym('q0_dot')
    q1_dot = SX.sym('q1_dot')
    q2_dot = SX.sym('q2_dot')
    q3_dot = SX.sym('q3_dot')
    omegax_dot = SX.sym('omegax_dot')
    omegay_dot = SX.sym('omegay_dot')
    omegaz_dot = SX.sym('omegaz_dot')

    xdot = vertcat(q0_dot, q1_dot, q2_dot, q3_dot, omegax_dot, omegay_dot, omegaz_dot)

    # algebraic variables
    # z = None

    # parameters
    p = []
    # This comes from ref[6] pg 449.
    # dq = 1/2 * G(q)' * Ω
    S = SX.zeros(3, 4)
    S[:, 0] = vertcat(-q1, -q2, -q3)
    S[:, 1:4] = q0*np.eye(3) - skew(vertcat(q1, q2, q3))

    print("S:", S, S.shape)

    OMG = vertcat(omegax, omegay, omegaz)
    # dynamics
    f_expl = vertcat(
                        0.5*mtimes(S.T,OMG),
                        u
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


    # Reference generation (quaternion to eul) used in cost function.
    Eul = SX(3,1)
    Eul[0, 0] = arctan2(2 * (q0*q1 + q2*q3), 1 - 2 * (q1**2 + q2**2))
    Eul[1, 0] = arcsin(2 * (q0*q2 - q3*q1))
    Eul[2, 0] = arctan2(2 * (q0*q3 + q1*q2), 1 - 2 * (q2**2 + q3**2))
    quat_to_eul = Function("quat_to_eul", [x, u], [Eul])

    model.cost_y_expr = vertcat(Eul, x, u)  #: CasADi expression for nonlinear least squares
    model.cost_y_expr_e = vertcat(Eul, x)  #: CasADi expression for nonlinear least squares, terminal

    model.con_h_expr = q0*q0 + q1*q1 + q2*q2 + q3*q3

    return model

