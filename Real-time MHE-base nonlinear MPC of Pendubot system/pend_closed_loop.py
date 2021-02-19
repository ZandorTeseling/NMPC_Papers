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

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver, AcadosModel
from export_pend_ocp_solver import export_ocp_solver
from export_pend_ode_model import export_pend_ode_model
from export_pend_mhe_solver import export_pend_mhe_solver
from utils import *
import numpy as nmp
from casadi import SX
import scipy.linalg

#NMPC controller cost model
COST_MODULE = 'NLS'

# set model
model = export_pend_ode_model()

ocp_model = export_pend_ode_model()
ocp_model.name = 'pend_ocp'

mhe_model = export_pend_ode_model()
mhe_model.name = 'pend_mhe'

sim_model = export_pend_ode_model()
sim_model.name = 'pend_sim'


#################################
### NMPC
#################################
# For optimal control
Tf = 1.0
Nmpc = 100
dt = Tf/Nmpc

nx = model.x.size()[0] - 2
nu = model.u.size()[0]
# model parameters
m1 = 0.265  # mass link 1
m2 = 0.226  # mass link 2
l1 = 0.206  # length link 1
l2 = 0.298  # length link 2
lc1 = 0.107  # cm link 1
lc2 = 0.133  # cm link 2
g = 9.81
p = nmp.array([m1, m2, l1, l2, lc1, lc2, g])

if COST_MODULE == 'NLS':
    # cos(q1) sin(q1) cos(q2) sin(q2) dq1 dq2 u
    Q = nmp.diag([1e3, 1e3, 1e3, 1e3, 1e0, 1e0])
elif COST_MODULE == 'LS':
    Q = nmp.diag([1e2, 1e2, 1, 1])

ContCost = 1.5e2 * nmp.ones(nu)
R = nmp.diag(ContCost)


b1 = 0.09  # damping coef joint 1 #0.08
b2 = 0.03  # damping coef joint 2 #0.00001
x0 = nmp.array([-1.5707,
               0.0,
               0.0,
               0.0,
               b1,  # damping coef joint 1
               b2    # damping coef joint 2
                ])
uMax = 2


acados_ocp_solver = export_ocp_solver(ocp_model, Nmpc, dt, Q, R, x0, uMax, COST_MODULE)


#################################
### MHE
#################################
Nmhe = 50
Qe = nmp.diag((1, 1))
Re = nmp.array(0.0001)
# Q0e = 10* nmp.eye(nx+2)
#x = q1 q2 dq1 dq2 b1 b2
Q0e = nmp.diag((1e0, 1e0, 2e0, 2e0, 1e3, 1e4))
acados_mhe_solver = export_pend_mhe_solver(mhe_model, Nmhe, dt, Qe, Q0e, Re)

#################################
### Dynamics Simulation
#################################
sim = AcadosSim()
sim.model = sim_model
sim.parameter_values = nmp.array([m1, m2, l1, l2, lc1, lc2, g])
# set simulation time
sim.solver_options.T = dt
# set options
sim.solver_options.num_stages = 4
sim.solver_options.num_steps = 6

acados_integrator = AcadosSimSolver(sim)

Tsim = 10
Nsim = int(Tsim/dt)

simX = nmp.zeros((Nsim+1, nx+2))
simXref = nmp.zeros((Nsim+1, 2))
simU = nmp.zeros((Nsim, nu))

v_stds = [0.05, 0.05]
simY = nmp.zeros((Nsim+1, 2)) #q1+w1 q2+w2
simXest = nmp.zeros((Nsim+1, nx + 2))
simUest = nmp.zeros((Nsim, 1))

# Initial conditions
simX[0, :] = x0

x0Bar = x0
x0Bar[4] = x0[4] + nmp.random.normal(0,0.05,1)
x0Bar[5] = x0[5] + nmp.random.normal(0,0.05,1)
for i in range(Nsim):
    print("Stage: ", i)
    # ~~~~~MHE~~~~~~~
    # get measurements
    simY[i, :] = simX[i, :2] + nmp.transpose(nmp.diag(v_stds) @ nmp.random.standard_normal((2, 1)))
    if i > Nmhe:
        for j in range(Nmhe):
            yref = nmp.zeros(2 + nx + nu + 2)
            yref[:2] = simY[(i-Nmhe)+j, :]
            yref[2] = simU[(i-Nmhe)+j, :] #+ 0.1 * nmp.random.standard_normal((1, 1))
            yref[3:] = x0Bar
            acados_mhe_solver.set(j, "yref", yref)
            acados_mhe_solver.set(j, "p", p)

        # solve mhe problem
        status = acados_mhe_solver.solve()

        # if status != 0:
            # raise Exception('acados returned status {}. Exiting.'.format(status))
        simUest[i, 0] = acados_mhe_solver.get(Nmhe-1, "u")

        simXest[i, :] = acados_mhe_solver.get(Nmhe, "x")
        x0Bar = simXest[i, :]

    # update state from true state for time being
    x0 = simX[i, :]

    # ~~~~~NMPC~~~~~~
    # solve ocp
    acados_ocp_solver.set(0, "x", x0)
    acados_ocp_solver.set(0, "lbx", x0)
    acados_ocp_solver.set(0, "ubx", x0)
    # update params
    for j in range(Nmpc):
         acados_ocp_solver.set(j, "p", p)


    # update trajectory
    t0 = i * dt


    for j in range(Nmpc):
        tCurr = t0 + j * dt
        if tCurr <= 10:
            # q1 down, q2 up
            if COST_MODULE == 'NLS':
                q1 = -nmp.pi/2
                q2 = nmp.pi
                refVec =nmp.array([nmp.cos(q1), nmp.sin(q1),
                                   nmp.cos(q2), nmp.sin(q2),
                                   0, 0, 0])
                acados_ocp_solver.cost_set(j, "y_ref", refVec)
                acados_ocp_solver.cost_set(Nmpc, "y_ref", refVec[:-1])
                acados_ocp_solver.cost_set(j, "W", nmp.diag([1e2, 1e2, 1e2, 1e2, 1e0, 1e0, 1.5e2]))
            # elif COST_MODULE == 'LS':
            #     acados_ocp_solver.set(j, "y_ref", nmp.array([1.5707, 0, 0, 0, 0]))
            #     # acados_ocp_solver.cost_set(j, "W", nmp.diag([1e2, 1e2, 1, 1, 1.5e2]))
            # acados_ocp_solver.cost_set(Nmpc, "W", nmp.diag([1e2, 1e2, 1e2, 1e2, 1e0, 1e0]))
        elif 10 < tCurr <= 20:
            # q1 up, q2 down
            if COST_MODULE == 'NLS':
                q1 = nmp.pi/2
                q2 = nmp.pi
                refVec =nmp.array([nmp.cos(q1), nmp.sin(q1),
                                   nmp.cos(q2), nmp.sin(q2),
                                   0, 0, 0])
                acados_ocp_solver.cost_set(j, "y_ref", refVec)
                acados_ocp_solver.cost_set(Nmpc, "y_ref", refVec[:-1])
                acados_ocp_solver.cost_set(j, "W", nmp.diag([1e2, 1e2, 1e2, 1e2, 1e0, 1e0, 1.0e1]))
            # elif COST_MODULE == 'LS':
            #     acados_ocp_solver.set(j, "y_ref",  nmp.array([1.5707, 0, 0, 0, 0]))
            # acados_ocp_solver.cost_set(Nmpc, "W", nmp.diag([1e2, 1e2, 1e2, 1e2, 1e0, 1e0]))
        elif 20 < tCurr <= 30:
            # q1 up, q2 up
            if COST_MODULE == 'NLS':
                q1 = nmp.pi / 2
                q2 = 0
                refVec = nmp.array([nmp.cos(q1), nmp.sin(q1),
                                    nmp.cos(q2), nmp.sin(q2),
                                    0, 0, 0])
                acados_ocp_solver.cost_set(j, "y_ref", refVec)
                acados_ocp_solver.cost_set(Nmpc, "y_ref", refVec[:-1])
                acados_ocp_solver.cost_set(j, "W", nmp.diag([1e2, 1e2, 1e2, 1e2, 1e0, 1e0, 4.0e1]))
                # acados_ocp_solver.cost_set(Nmpc, "W", nmp.diag([1e2, 1e2, 1e2, 1e2, 1e1, 1e1]))
            # elif COST_MODULE == 'LS':
            #     acados_ocp_solver.set(j, "y_ref",  nmp.array([1.5707, 0, 0, 0, 0]))

        elif 30 < tCurr <= 40:
            # q1 down, q2 down
            if COST_MODULE == 'NLS':
                q1 = -nmp.pi / 2
                q2 = 0
                refVec = nmp.array([nmp.cos(q1), nmp.sin(q1),
                                    nmp.cos(q2), nmp.sin(q2),
                                    0, 0, 0])
                acados_ocp_solver.cost_set(j, "y_ref", refVec)
                acados_ocp_solver.cost_set(Nmpc, "y_ref", refVec[:-1])
                acados_ocp_solver.cost_set(j, "W", nmp.diag([1e2, 1e2, 1e2, 1e2, 1e0, 1e0, 3.0e1]))
                # acados_ocp_solver.cost_set(Nmpc, "W", nmp.diag([1e2, 1e2, 1e2, 1e2, 1e0, 1e0]))
            # elif COST_MODULE == 'LS':
            #     acados_ocp_solver.set(j, "y_ref",  nmp.array([1.5707, 0, 0, 0, 0]))


    status = acados_ocp_solver.solve()
    if status != 0 and status != 2:
        raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))

    simU[i, :] = acados_ocp_solver.get(0, "u")
    # ~~~~~'Real' System~~~~~~
    acados_integrator.set("x", x0)
    acados_integrator.set("u", simU[i, :])
    status = acados_integrator.solve()
    if status != 0:
        raise Exception('acados integrator returned status {}. Exiting.'.format(status))

    simXref[i, :] = nmp.array([q1,q2])
    simX[i + 1, :] = acados_integrator.get("x") #True state.

# plot results
plot_double_pendulum(dt, uMax, simU, simX, simXref, simXest, simUest, simY)

