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
import numpy as nmp
import scipy.linalg


COST_MODULE = 'NLS'
USE_QUAT_SLACK = 0
# create ocp object to formulate the OCP
ocp = AcadosOcp()

# set model
model = export_quad_ode_model()
ocp.model = model

# model parameters
rho =   1.225
A =     0.1
Cl =    0.125
Cd =    0.075
m =     10.0
g =     9.81
J3 =    0.25
J2 =    J3*4
J1 =    J3*4
p = nmp.array([rho, A, Cl, Cd, m, g, J1, J2, J3])

#For optimal control
Tf = 1.0
N = 20
dt = Tf/N

nx = model.x.size()[0]
nu = model.u.size()[0]
# nz  = model.z.size()[0]
np = model.p.size()[0]
if USE_QUAT_SLACK == 1:
    nh = model.con_h_expr.size()[0]

AttCost = 5 * (10**2) * nmp.ones(3)
StateCost = 1 * (10**-3) * nmp.ones(nx)

if COST_MODULE == 'LS':
    # set cost module
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ny = nx + nu
    ny_e = nx
    ocp.dims.ny = ny
    ocp.dims.ny_e = ny_e
    # W = [ 1*10^-3* I_11]
    # Wn = [1*q0^-3 * I_7]

    ocp.cost.Vx = nmp.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = nmp.eye(nx)

    ocp.cost.Vu = nmp.zeros((ny, nu))
    ocp.cost.Vu[nx:, :nu] = nmp.eye(nu)
    ocp.cost.Vx_e = nmp.eye(nx)
    Q = nmp.diag((StateCost))

elif COST_MODULE == 'NLS':
    # From the paper
    # W = [5*10^2*I_3, 1*10^-3 I_11]
    # Wn = [5*10^2*I_3, 1*q0^-3 * I_7]

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ny = nx + nu + 3 # plus Attitude
    ny_e = nx + 3 # plus Attitude
    ocp.dims.ny = ny
    ocp.dims.ny_e = ny_e

    Q = nmp.diag(nmp.concatenate((AttCost, StateCost)))

if USE_QUAT_SLACK == 1:
    ocp.dims.nh = nh
    ocp.dims.nsh = 1
    ocp.dims.ns = 1
    # cost on slack variable
    ocp.cost.Zl = nmp.array([10])
    ocp.cost.Zu = nmp.array([10])
    ocp.cost.zl = nmp.array([10])
    ocp.cost.zu = nmp.array([10])

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

# Due to the paper using the forulation in cascade,
# some steady state is assumed as reference form a position controller.
uSS = 39.99
if COST_MODULE == 'LS':
    # q0 q1 q2 q3 omeg1 omeg2 omeg3 w1 w2 w3 w4
    ocp.cost.yref  = nmp.zeros((nx+nu, ))
    ocp.cost.yref[0] = 1 # Quaternion
    ocp.cost.yref[nx:] = uSS
    # q0 q1 q2 q3 omeg1 omeg2 omeg3
    ocp.cost.yref_e = nmp.zeros((nx, ))
    ocp.cost.yref_e[0] = 1
elif COST_MODULE == 'NLS':
    # roll pitch yaw q0 q1 q2 q3 omeg1 omeg2 omeg3 w1 w2 w3 w4
    ocp.cost.yref  = nmp.zeros((14, ))
    ocp.cost.yref[3] = 1# Quaternion
    ocp.cost.yref[nx+3:] = uSS
    # roll pitch yaw q0 q1 q2 q3 omeg1 omeg2 omeg3
    ocp.cost.yref_e = nmp.zeros((10, ))
    ocp.cost.yref_e[3] = 1


# init conditions, start at yaw of +90deg
x0 = nmp.array([0.285,
               -0.959,
               0.0,
               0.0,
               0.0,
               0.0,
               0.0])

# set constraints
uDel = 8
uSS = 39.99
ocp.constraints.constr_type = 'BGH'
ocp.constraints.lbu = nmp.array([uSS-uDel, uSS-uDel, uSS-uDel, uSS-uDel])
ocp.constraints.ubu = nmp.array([uSS+uDel, uSS+uDel, uSS+uDel, uSS+uDel])
ocp.constraints.x0 = x0
ocp.parameter_values = p
ocp.constraints.idxbu = nmp.array([0, 1, 2, 3])

if USE_QUAT_SLACK == 1:
    # nonlinear quaternion constraint
    ocp.constraints.lh = nmp.array([1.0])
    ocp.constraints.uh = nmp.array([1.0])
    # #slack for nonlinear quat
    ocp.constraints.lsh = nmp.array([-0.2])
    ocp.constraints.ush = nmp.array([0.2])
    ocp.constraints.idxsh = nmp.array([0])

ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'#'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.integrator_type = 'IRK'
ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI
ocp.solver_options.print_level = 0

# set prediction horizon
ocp.solver_options.tf = Tf

acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')

acados_integrator = AcadosSimSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')
acados_integrator.set("p", p)

Nsim = 200

simX = nmp.zeros((Nsim+1, nx))
simU = nmp.zeros((Nsim, nu))

simX[0, :] = x0

for i in range(Nsim):
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
        if tCurr <= 2:
            # roll = 1 pitch = -1 yaw = 0
            # q = 0.770 0.421 -0.421 0.230
            acados_ocp_solver.set(j, "y_ref", nmp.array([1, -1, 0, 0.770, 0.421, -0.421, 0.230, 0, 0, 0, uSS, uSS, uSS, uSS]))
        elif tCurr <= 4:
            # roll = -1 pitch = -1 yaw = 0
            # q = 0.770 -0.421 -0.421 -0.230
            acados_ocp_solver.set(j, "y_ref",  nmp.array([-1, -1, 0, 0.770, -0.421, -0.421, -0.230, 0, 0, 0, uSS, uSS, uSS, uSS]))
        elif tCurr <= 6:
            # roll = -1 pitch = 1 yaw = 0
            # q = 0.770 -0.421 0.421 0.230
            acados_ocp_solver.set(j, "y_ref",  nmp.array([-1, 1, 0, 0.770, -0.421, 0.421, 0.230, 0, 0, 0, uSS, uSS, uSS, uSS]))

    status = acados_ocp_solver.solve()
    if status != 0:
        raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))

    simU[i, :] = acados_ocp_solver.get(0, "u")
    acados_integrator.set("x", x0)
    acados_integrator.set("u", simU[i, :])
    status = acados_integrator.solve()
    if status != 0:
        raise Exception('acados integrator returned status {}. Exiting.'.format(status))

    # update state
    x0 = acados_integrator.get("x")
    simX[i+1, :] = x0

qtest = YPRtoQuat(nmp.array([nmp.pi/4, nmp.pi/4, 0]))
print("q\n",qtest)
R = QuattoR(qtest)
print("RMat:\n", R)
# plot results
plot_quad(dt, uSS, uDel, simU, simX)

