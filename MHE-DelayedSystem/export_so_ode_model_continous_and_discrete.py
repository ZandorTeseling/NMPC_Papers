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

def export_so_ode_ct():

    model_name = 'second_order_ode_ct'

    # set up states & controls
    z1 = SX.sym('z1')
    z2 = SX.sym('z2')
    x = vertcat(z1, z2)

    # control
    u = SX.sym('u')

    # xdot
    z1_dot = SX.sym('z1_dot')
    z2_dot = SX.sym('z2_dot')

    xdot = vertcat(z1_dot, z2_dot)

    # algebraic variables
    # z = None

    # parameters
    # set up parameters
    zeta = SX.sym('zeta')   # zeta
    ts   = SX.sym('ts')  # ts
    Kp   = SX.sym('Kp')  # Kp

    p = vertcat(zeta, ts, Kp)

    # dynamics
    f_expl = vertcat(
                    z2,
                    -1/(ts*ts)*z1 - 2*zeta/ts*z2 + Kp/(ts*ts)*u
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

def export_so_ode_dt_rk4(dT):
    inputDelay = 0.553

    model = export_so_ode_ct()
    model.name = model_name = 'second_order_ode_dt_rk4'

    x = model.x
    u = model.u
    u_in = SX.sym('u_in')

    Nx = x.size()[0]

    Naug = int(floor(inputDelay/dT)) #Number of control input propegation delay
    u_aug = SX.sym('u_aug',Naug)
    Aaug = SX.zeros(Naug, Naug + Nx)
    #Fill superdiagonal with 1
    for i in range(0, Naug-1):
        for j in range(Nx, Naug+Nx):
            if(j-Nx == i+1):
                Aaug[i,j] = 1

    print("Aaug:" , Aaug)
    Baug = SX.zeros(Naug, 1)
    Baug[-1, 0] = 1
    print("Baug:", Baug)

    x_aug = vertcat(x,u_aug)
    print("Aaug*x+Baug*u:",  mtimes(Aaug, x_aug) + mtimes(Baug, u_in))


    model.name = 'augmented_second_order_ode_dt'
    model.f_expl_expr = substitute(model.f_expl_expr, u, u_aug[0])

    ode = Function('ode', [x, u_aug[0]], [model.f_expl_expr])
    # set up RK4
    k1 = ode(x, u_aug[0])
    k2 = ode(x+dT/2*k1, u_aug[0])
    k3 = ode(x+dT/2*k2, u_aug[0])
    k4 = ode(x+dT*k3, u_aug[0])
    xrk4 = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

    print("ode: ", ode)
    print("rk4: ", xrk4)
    xf = vertcat(xrk4, mtimes(Aaug, x_aug) + mtimes(Baug, u_in))
    model.disc_dyn_expr = xf
    model.x = x_aug
    model.u = u_in

    print("built RK4 model with dT = ", dT)
    print(xf)
    return model