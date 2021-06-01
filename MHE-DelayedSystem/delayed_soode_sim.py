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

from acados_template import AcadosSim, AcadosSimSolver
from export_so_ode_model_continous_and_discrete import export_so_ode_ct, export_so_ode_dt_rk4
from utils import *
from casadi import *
import numpy as nmp
import matplotlib.pyplot as plt

dt = 0.05
Tf = 10.0
N = int(Tf/dt)

sim_ct = AcadosSim()
sim_dt = AcadosSim()
# export model
model_ct = export_so_ode_ct()
zeta = 0.5 #Under Damped
ts   = 0.125 #Time Constant
Kp   = 1.0 #Steady State Gain

inputDelay = 0.553
model_dt = export_so_ode_dt_rk4(dt, inputDelay)

sim_ct.model = model_ct
sim_dt.model = model_dt

sim_ct.parameter_values = np.array([zeta, ts, Kp])
sim_dt.parameter_values = np.array([zeta, ts, Kp, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# set simulation time
sim_ct.solver_options.T = dt
sim_dt.solver_options.T = dt
# set options
sim_ct.solver_options.integrator_type = 'ERK'
sim_ct.solver_options.num_stages = 4
sim_ct.solver_options.num_steps = 3
sim_ct.solver_options.newton_iter = 3 # for implicit integrator

sim_dt.solver_options.integrator_type = 'DISCRETE'
# create
acados_integrator_ct = AcadosSimSolver(sim_ct) #Used to mock data for testing mhe
acados_integrator_dt = AcadosSimSolver(sim_dt) #Used to mock data for testing mhe

nx = model_ct.x.size()[0]
nx_dt = model_dt.x.size()[0]
nu = model_ct.u.size()[0]
u0 = nmp.zeros(nu)

#Storage structs
simX = nmp.zeros((N+1, nx))
simXdisc = nmp.zeros((N+1, nx_dt))
simU = nmp.zeros((N, nu))
simY = nmp.zeros((N+1, 1))
simYdisc = nmp.zeros((N+1, 1))

#inital state/parameters
x0 = nmp.zeros(nx)
x0disc = nmp.zeros(nx_dt)

########################
# ode used to generate data to test.
########################
simX[0, :] = x0
simXdisc[0, :] = x0disc
v_stds = [0.05]

simU[round(0.8/dt): round(3.8/dt), 0] =  1
simU[round(5.8/dt): round(7.8/dt), 0] = -1

for i in range(N):
    # set initial state
    u0 = simU[i, :]
    p = nmp.array([zeta, ts, Kp])

    # Pick which control input actuates the body, allows for varying time delay.
    p_dt = nmp.array([zeta, ts, Kp, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    acados_integrator_ct.set("p", p)
    acados_integrator_ct.set("x", x0)
    acados_integrator_ct.set("u", u0)

    acados_integrator_dt.set("p", p_dt)
    acados_integrator_dt.set("x", x0disc)
    acados_integrator_dt.set("u", u0)

    status = acados_integrator_dt.solve()
    if status != 0:
        raise Exception('acados returned status {}. Exiting.'.format(status))
    # get solution
    x0disc = acados_integrator_dt.get("x")

    status = acados_integrator_ct.solve()
    if status != 0:
        raise Exception('acados returned status {}. Exiting.'.format(status))
    # get solution
    x0 = acados_integrator_ct.get("x")


    simXdisc[i + 1, :] = x0disc
    simX[i + 1, :]     = x0

    simY[i+1, :] = x0[0] + nmp.transpose(nmp.diag(v_stds) @ nmp.random.standard_normal((1, 1)))
    simYdisc[i + 1, :] = x0disc[0] + nmp.transpose(nmp.diag(v_stds) @ nmp.random.standard_normal((1, 1)))

plt.figure(1)
plt.plot(np.linspace(0, Tf, N+1), simX[:, 0], label="true no delay")
plt.step(np.linspace(0, Tf, N+1), simXdisc[:, 0], label="true disc delayed")
plt.plot(np.linspace(0, Tf, N), simU[:, 0], label="input")
plt.plot(np.linspace(0, Tf, N+1), simY[:, 0], 'x', label="measured ")
plt.plot(np.linspace(0, Tf, N+1), simYdisc[:, 0], '.', label="measured disc")

plt.ylabel("[-]")
plt.xlabel("time [s]")
plt.title("SO+DeadTime ODE Dynamic Model with InputDelay: " + str(inputDelay) + " tS: " + str(ts) + " Kp: " + str(Kp) + " Zeta: " + str(zeta))
plt.grid()
plt.legend(loc="upper left")
plt.show(block = False)

plt.show()

