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

from acados_template import AcadosOcp, AcadosOcpSolver
from export_quaternion_ode_model import export_quaternion_ode_model
import numpy as nmp
import scipy.linalg
from utils import plot_quad


COST_MODULE = 'NLS' # 'LS', 'EXTERNAL'
USE_SLACK = 1

# create ocp object to formulate the OCP
ocp = AcadosOcp()

# set model
model = export_quaternion_ode_model()
ocp.model = model

Tf = 2.0
nx = model.x.size()[0]
nu = model.u.size()[0]
nh = model.con_h_expr.size()[0]
if COST_MODULE == 'LS':
    ny = model.x.size()[0] + model.u.size()[0]
    ny_e = model.x.size()[0]
    ocp.dims.ny = ny
    ocp.dims.ny_e = ny_e

    ocp.cost.Vx = nmp.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = nmp.eye(nx)

    Vu = nmp.zeros((ny, nu))
    Vu[nx:ny, :] = nmp.eye(nu)
    ocp.cost.Vu = Vu
    ocp.cost.Vx_e = nmp.eye(nx)

    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

elif COST_MODULE == 'NLS':
    ny = model.cost_y_expr.size()[0]
    ny_e = model.cost_y_expr_e.size()[0]
    ocp.dims.ny = ny
    ocp.dims.ny_e = ny_e

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

N = 40


# set type independent dimensions
ocp.dims.nx = nx
if USE_SLACK:
    ocp.dims.nh = nh
    ocp.dims.nsh = 1
    ocp.dims.ns = 1

ocp.dims.nbu = nu
ocp.dims.nu = nu
ocp.dims.N = N

# set cost
if COST_MODULE == 'LS':
    #q0 q1 q2 q3 omegx omegy omegz
    Q = 2 * nmp.diag([1e3, 1e3, 1e3, 1e3, 1e-3, 1e-3, 1e-3])
    #w1 w2 w3
    R = 2 * nmp.diag([1e-4, 1e-4, 1e-4])
    ocp.cost.yref  = nmp.zeros((ocp.dims.ny, ))
    ocp.cost.yref[0] = 1
    ocp.cost.yref_e = nmp.zeros((ocp.dims.ny_e, ))
    ocp.cost.yref_e[0] = 1
elif COST_MODULE == 'NLS':
    # r p y q0 q1 q2 q3 omegx omegy omegz
    Q = 2*nmp.diag([1e-3, 1e-3, 1e-3, 1e3, 1e3, 1e3, 1e3, 1e-4, 1e-4, 1e-4])
    #dw1 dw2 dw3
    R = 2*nmp.diag([1e-8, 1e-8, 1e-8])
    ocp.cost.yref  = nmp.zeros((ocp.dims.ny, ))
    ocp.cost.yref[3] = 1
    ocp.cost.yref_e = nmp.zeros((ocp.dims.ny_e, ))
    ocp.cost.yref_e[3] = 1

# cost on slack variable
if USE_SLACK:
    ocp.cost.Zl = nmp.array([1e2])
    ocp.cost.Zu = nmp.array([1e2])
    ocp.cost.zl = nmp.array([1e2])
    ocp.cost.zu = nmp.array([1e2])

ocp.cost.W_e = Q
ocp.cost.W = scipy.linalg.block_diag(Q, R)

# set constraints
# initial conditions constraints q0 q1 q2 q3 omegx omegy omegz
ocp.constraints.x0 = nmp.array([0.707, 0.707, 0.0, 0.0, 0.0, 0.0, 0.0])

# control constraints
# 'BGH' Comprises simple bounds, polytopic constraints, general non-linear constraints.
# 'BGP' Comprises simple bounds, polytopic constraints, general non-linear constraints, and positive definite constraints.
ocp.constraints.constr_type = 'BGH'


omegMax = 10.0
ocp.constraints.lbu = nmp.array([-0.5*omegMax, -0.75*omegMax, -omegMax])
ocp.constraints.ubu = nmp.array([0.5*omegMax, 0.75*omegMax, omegMax])
ocp.constraints.idxbu = nmp.array([0, 1, 2])

# nonlinear quaternion constraint
if USE_SLACK:
    ocp.constraints.lh = nmp.array([1.0])
    ocp.constraints.uh = nmp.array([1.0])
    # slack for nonlinear quaternion
    ocp.constraints.lsh = nmp.array([-1e-2])
    ocp.constraints.ush = nmp.array([1e-2])
    ocp.constraints.idxsh = nmp.array([0])


ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.integrator_type = 'IRK'

# set prediction horizon
ocp.solver_options.tf = Tf
ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI
ocp.solver_options.print_level = 0
ocp.solver_options.qp_solver_iter_max = 200
ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')

simX = nmp.ndarray((N+1, nx))
simU = nmp.ndarray((N, nu))

status = ocp_solver.solve()

# if status != 0:
#     raise Exception('acados returned status {}. Exiting.'.format(status))

# get solution
for i in range(N):
    simX[i,:] = ocp_solver.get(i, "x")
    simU[i,:] = ocp_solver.get(i, "u")
simX[N,:] = ocp_solver.get(N, "x")

plot_quad(Tf/N, omegMax, simU, simX)
