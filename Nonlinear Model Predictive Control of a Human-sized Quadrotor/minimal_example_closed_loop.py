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
from export_quad_ode_model import export_quad_ode_model
from utils import *
import numpy as np
import scipy.linalg

# create ocp object to formulate the OCP
ocp = AcadosOcp()

# set model
model = export_quad_ode_model()
ocp.model = model

Tf = 1.0
Dt = 0.05
nx = model.x.size()[0]
nu = model.u.size()[0]
ny = nx + nu + 3
ny_e = nx + 3
N = 20

# set dimensions
ocp.dims.nx  = nx
ocp.dims.ny  = ny
ocp.dims.ny_e = ny_e
# ocp.dims.nbu = nu
ocp.dims.nu  = nu
ocp.dims.N   = N


# set cost module
ocp.cost.cost_type = 'NONLINEAR_LS'
ocp.cost.cost_type_e = 'NONLINEAR_LS'

# From the paper
# W = [5*10^2*I_3, 1*10^-3 I_11]
# Wn = [5*10^2*I_3, 1*q0^-3 * I_7]
AttCost = 5 * (10**2) * np.ones(3)
StateCost = 1 * (10**-3) * np.ones(nx)
Q = np.diag(np.concatenate((AttCost, StateCost)))

ContCost = 1 * (10**-3) * np.ones(nu)
R = np.diag(ContCost)

ocp.cost.W = scipy.linalg.block_diag(Q, R)
ocp.cost.W_e = Q

ocp.cost.yref  = np.zeros((14, ))
ocp.cost.yref[3] = 1
ocp.cost.yref_e = np.zeros((10, ))
ocp.cost.yref_e[3] = 1

# init conditions
x0 = np.array([1.0,
               0.0,
               0.0,
               0.0,
               0.0,
               0.0,
               0.0])

# set constraints
uMax = 39.99
ocp.constraints.constr_type = 'BGH'
# ocp.constraints.lbu = np.array([-uMax, -uMax, -uMax, -uMax])
# ocp.constraints.ubu = np.array([uMax, uMax, uMax, uMax])
ocp.constraints.x0 = x0
# ocp.constraints.idxbu = np.array([0, 1, 2, 3])

ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'#'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI
ocp.solver_options.qp_solver_iter_max = 100

# ocp.solver_options.qp_solver_cond_N = N

# set prediction horizon
ocp.solver_options.tf = Tf

acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')
acados_integrator = AcadosSimSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')

#Parameters
rho =   1.225
A =     0.1
Cl =    0.125
Cd =    0.075
m =     10.0
g =     9.81
J3 =    0.25
J2 =    J3*4
J1 =    J3*4
p = np.array([rho, A, Cl, Cd, m, g, J1, J2, J3])
acados_integrator.set("p", p)


simX = np.ndarray((N+1, nx))
simU = np.ndarray((N, nu))

xcurrent = x0
simX[0, :] = xcurrent

print(xcurrent)
# closed loop
for i in range(N):
    # solve ocp
    acados_ocp_solver.set(0, "lbx", xcurrent)
    acados_ocp_solver.set(0, "ubx", xcurrent)
    # update params
    for j in range(N):
        acados_ocp_solver.set(j, "p", p)

    status = acados_ocp_solver.solve()
    if status != 0:
        raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))

    simU[i,:] = acados_ocp_solver.get(0, "u")
    print(simU[i,:])
    # simulate system
    acados_integrator.set("x", xcurrent)
    acados_integrator.set("u", simU[i,:])
    print(xcurrent)
    status = acados_integrator.solve()
    if status != 0:
        raise Exception('acados integrator returned status {}. Exiting.'.format(status))

    # update state
    xcurrent = acados_integrator.get("x")
    simX[i+1, :] = xcurrent

# plot results
plot_quad(Tf/N, uMax, simU, simX)
