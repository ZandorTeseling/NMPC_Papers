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

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from export_pend_ocp_solver import export_ocp_solver
from export_pend_ode_model import export_pend_ode_model
from utils import *
import numpy as nmp
import scipy.linalg


COST_MODULE = 'LS'
# create ocp object to formulate the OCP
ocp = AcadosOcp()

# set model
model = export_pend_ode_model()
ocp.model = model

# model parameters
m1 = 0.265  # mass link 1
m2 = 0.226  # mass link 2
l1 = 0.206  # length link 1
l2 = 0.298  # length link 2
lc1 = 0.107  # cm link 1
lc2 = 0.133  # cm link 2
g = 9.81
p = nmp.array([m1, m2, l1, l2, lc1, lc2, g])

# For optimal control
Tf = 2.0
N = 50
dt = Tf/N

nx = model.x.size()[0] - 2
nu = model.u.size()[0]
# nz  = model.z.size()[0]
np = model.p.size()[0]

StateCost = nmp.ones(nx)
StateCost[0] = 1e3
StateCost[1] = 1e3

if COST_MODULE == 'LS':
    # set cost module
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ny = nx + nu
    ny_e = nx
    ocp.dims.ny = ny
    ocp.dims.ny_e = ny_e

    ocp.cost.Vx = nmp.zeros((ny, nx+2))
    ocp.cost.Vx[:nx, :nx] = nmp.eye(nx)

    ocp.cost.Vu = nmp.zeros((ny, nu))
    ocp.cost.Vu[nx:, :nu] = nmp.eye(nu)

    ocp.cost.Vx_e = nmp.zeros((ny_e, nx + 2))
    ocp.cost.Vx_e[:ny_e, :ny_e] = nmp.eye(ny_e)
    Q = nmp.diag((StateCost))

print("Q:\n", Q)
print("Vx:\n", ocp.cost.Vx)
print("Vx_e:\n", ocp.cost.Vx_e)
print("Vu:\n", ocp.cost.Vu)
# set cost module independent dimensions
ocp.dims.nx = nx
ocp.dims.nbu = nu
ocp.dims.nu = nu
ocp.dims.np = np
ocp.dims.N = N

ContCost = 1 * (10**-3) * nmp.ones(nu)
R = nmp.diag(ContCost)

ocp.cost.W = scipy.linalg.block_diag(Q, R)
ocp.cost.W_e = Q

if COST_MODULE == 'LS':
    ocp.cost.yref  = nmp.zeros((nx+nu, ))
    ocp.cost.yref[0] = 1.5707 # joint 1 u
    ocp.cost.yref_e = nmp.zeros((nx, ))
    ocp.cost.yref_e[0] = 1.5707


b1 = 0.2  # damping coef joint 1
b2 = 0.1  # damping coef joint 2
x0 = nmp.array([-1.5707,
               0.0,
               0.0,
               0.0,
               b1,  # damping coef joint 1
               b2    # damping coef joint 2
                ])

# set constraints
uMax = 4.0
ocp.constraints.constr_type = 'BGH'
ocp.constraints.lbu = nmp.array([-uMax])
ocp.constraints.ubu = nmp.array([uMax])
ocp.constraints.x0 = x0
ocp.parameter_values = p
ocp.constraints.idxbu = nmp.array([0 ])

ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'#'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI
# ocp.solver_options.nlp_solver_max_iter = 5
ocp.solver_options.print_level = 0

# set prediction horizon
ocp.solver_options.tf = Tf

# acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')
acados_ocp_solver = export_ocp_solver(model, N, dt, Q, R, uMax)

acados_integrator = AcadosSimSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')
acados_integrator.set("p", p)

Nsim = int(40/dt)

simX = nmp.zeros((Nsim+1, nx+2))
simU = nmp.zeros((Nsim, nu))

simX[0, :] = x0

for i in range(Nsim):
    # ~~~~~NMPC~~~~~~
    # solve ocp
    acados_ocp_solver.set(0, "lbx", x0)
    acados_ocp_solver.set(0, "ubx", x0)
    # update params
    for j in range(N):
        acados_ocp_solver.set(j, "p", p)

    # update trajectory
    t0 = i * dt
    for j in range(N):
        tCurr = t0 + j * dt
        if tCurr <= 20:
            # j1 up, j2 up
            acados_ocp_solver.set(j, "y_ref",  nmp.array([1.5707, 0, 0, 0, 0]))
        elif tCurr <= 40:
            # j1 up, j2 down
            acados_ocp_solver.set(j, "y_ref",  nmp.array([1.5707, 3.14, 0, 0, 0]))

    status = acados_ocp_solver.solve()
    # if status != 0:
    #     raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))

    simU[i, :] = acados_ocp_solver.get(0, "u")
    # simulate real system.
    acados_integrator.set("x", x0)
    acados_integrator.set("u", simU[i, :])
    status = acados_integrator.solve()
    if status != 0:
        raise Exception('acados integrator returned status {}. Exiting.'.format(status))

    # ~~~~~MHE~~~~~~~
    # get measurements
    # update state
    x0 = acados_integrator.get("x")
    simX[i+1, :] = x0

# plot results
plot_double_pendulum(dt, uMax, simU, simX)

