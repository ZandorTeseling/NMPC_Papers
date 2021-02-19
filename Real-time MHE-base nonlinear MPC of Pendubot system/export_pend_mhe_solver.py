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
from scipy.linalg import block_diag
from acados_template import *

def export_pend_mhe_solver(model, N, h, Q, Q0, R):

    print("Q:", Q)
    print("Q0:", Q0)
    print("R:", R)
    # create render arguments
    ocp_mhe = AcadosOcp()
    ocp_mhe.model = model

    #q1 q2 dq1 dq2 b1 b2
    nx = model.x.size()[0]
    #tau1
    nu = model.u.size()[0]
    # m1 m2 l1 l2 lc1 lc2 g
    np = model.p.size()[0]

    # In the formulation of this paper, they assume that there
    # is noise on the control and measurement noise.
    # y_ref  = q1, q2, u, x0

    #h(x) = [q1, q2]
    ny = 2 + nu + nx  # h(x),u,x0
    ny_e = 0

    #ocp dimensions
    ocp_mhe.dims.nx = nx
    ocp_mhe.dims.ny = ny
    ocp_mhe.dims.np = np
    ocp_mhe.dims.ny_e = ny_e
    ocp_mhe.dims.nbx = 2
    ocp_mhe.dims.nbu = 0
    ocp_mhe.dims.nu  = model.u.size()[0]
    ocp_mhe.dims.N   = N
    # No initial conditions constraint
    ocp_mhe.dims.nbx_0 = 0

    ocp_mhe.cost.cost_type = 'NONLINEAR_LS'
    ocp_mhe.cost.cost_type_e = 'LINEAR_LS'

    ocp_mhe.cost.W = block_diag(Q, R, nmp.zeros((nx, nx)))

    ocp_mhe.model.cost_y_expr = vertcat(ocp_mhe.model.x[0],
                                        ocp_mhe.model.x[1],
                                        ocp_mhe.model.u,
                                        ocp_mhe.model.x)
    print( vertcat(ocp_mhe.model.x[0],
                                        ocp_mhe.model.x[1],
                                        ocp_mhe.model.u,
                                        ocp_mhe.model.x) )
    ocp_mhe.parameter_values = nmp.zeros((np, ))

    print("Ref size ny:", ny)
    ocp_mhe.cost.yref  = nmp.zeros((ny,))
    ocp_mhe.cost.yref_e = nmp.zeros((ny_e, ))

    ocp_mhe.constraints.ubx = nmp.array([5, 5])
    ocp_mhe.constraints.lbx = nmp.array([0, 0])
    ocp_mhe.constraints.idxbx = nmp.array([4, 5])

    # set QP solver
    ocp_mhe.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    # ocp_mhe.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp_mhe.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp_mhe.solver_options.integrator_type = 'ERK'

    # set prediction horizon
    ocp_mhe.solver_options.tf = N * h

    #ocp_mhe.solver_options.nlp_solver_type = 'SQP'
    ocp_mhe.solver_options.nlp_solver_type = 'SQP_RTI'
    #ocp_mhe.solver_options.nlp_solver_max_iter = 50

    acados_mhe_solver = AcadosOcpSolver(ocp_mhe, json_file='acados_mhe.json')

    # set arrival cost weighting matrix
    acados_mhe_solver.cost_set(0, "W", block_diag(Q, R, Q0))

    return acados_mhe_solver
