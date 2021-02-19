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

import numpy as nmp
import scipy
from acados_template import *

def export_ocp_solver(model, N, h, Q, R, x0, Taumax=2, COST_MODULE='LS' ):

    # create render arguments
    ocp = AcadosOcp()

    # set model
    ocp.model = model

    Tf = N*h
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    np = model.p.size()[0]


    # set cost
    if COST_MODULE == 'LS':
        # set cost module
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ny = nx + nu - 2
        ny_e = nx - 2
        ocp.dims.ny = ny
        ocp.dims.ny_e = ny_e

        ocp.cost.Vx = nmp.zeros((ny, nx ))
        ocp.cost.Vx[:nx-2, :nx-2] = nmp.eye(nx-2)

        ocp.cost.Vu = nmp.zeros((ny, nu))
        ocp.cost.Vu[nx-2:, :nu] = nmp.eye(nu)

        ocp.cost.Vx_e = nmp.zeros((ny_e, nx ))
        ocp.cost.Vx_e[:ny_e, :ny_e] = nmp.eye(ny_e)
    elif COST_MODULE == 'NLS':
        # set cost module
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ny = nx + nu - 2 + 2 # cos(q1) sin(q1) cos(q2) sin(q2) dq1 dq2 u:  no cost on b1 b2
        ny_e = nx - 2 + 2
        ocp.dims.ny = ny
        ocp.dims.ny_e = ny_e

        ocp.model.cost_y_expr = vertcat(cos(model.x[0]),
                                        sin(model.x[0]),
                                        cos(model.x[1]),
                                        sin(model.x[1]),
                                        model.x[2],
                                        model.x[3],
                                        model.u)
        ocp.model.cost_y_expr_e = vertcat(cos(model.x[0]),
                                        sin(model.x[0]),
                                        cos(model.x[1]),
                                        sin(model.x[1]),
                                        model.x[2],
                                        model.x[3])

    # set cost module independent dimensions
    ocp.dims.nx = nx -2
    ocp.dims.nbu = nu
    ocp.dims.nbx = 1
    ocp.dims.nu = nu
    ocp.dims.np = np
    ocp.dims.N = N

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q

    if COST_MODULE == 'LS':
        ocp.cost.yref = nmp.zeros((nx-2 + nu,))
        ocp.cost.yref[0] = 1.5707  # joint 1 u
        ocp.cost.yref_e = nmp.zeros((nx-2,))
        ocp.cost.yref_e[0] = 1.5707
    elif COST_MODULE == 'NLS':
        ocp.cost.yref = nmp.zeros((nx-2 + 2 + nu,))
        ocp.cost.yref[0] = cos(1.5707)
        ocp.cost.yref[1] = sin(1.5707)
        ocp.cost.yref_e = nmp.zeros((nx-2 + 2,))
        ocp.cost.yref_e[0] = cos(1.5707)
        ocp.cost.yref_e[1] = sin(1.5707)
    # setting bounds
    ocp.constraints.lbu = nmp.array([-Taumax])
    ocp.constraints.ubu = nmp.array([+Taumax])
    ocp.constraints.ubx =  nmp.array([2*nmp.pi])
    ocp.constraints.lbx =  nmp.array([-2*nmp.pi])
    ocp.constraints.x0 = x0

    ocp.constraints.idxbu = nmp.array([0])
    ocp.constraints.idxbx = nmp.array([0])
    ocp.parameter_values = nmp.zeros((np, ))

    # set QP solver
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'

    # set prediction horizon
    ocp.solver_options.tf = Tf
    ocp.solver_options.nlp_solver_type = 'SQP'
    # ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.nlp_solver_max_iter = 5
    ocp.solver_options.print_level = 0

    return AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')