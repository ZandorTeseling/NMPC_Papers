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
Tf = 5.0
N = int(Tf/dt)


sim_ct = AcadosSim()
sim_dt = AcadosSim()
# export model
model_ct = export_so_ode_ct()
zeta = 1.0 #Critically Damped
ts = 0.3 #Time Constant
Kp = 1.0 #Steady State Gain


model_dt = export_so_ode_dt_rk4(dt)

sim_ct.model = model_ct

sim_ct.parameter_values = np.array([zeta, ts, Kp])
# set simulation time
sim_ct.solver_options.T = dt
# set options
sim_ct.solver_options.integrator_type = 'ERK'
sim_ct.solver_options.num_stages = 4
sim_ct.solver_options.num_steps = 3
sim_ct.solver_options.newton_iter = 3 # for implicit integrator
# create
acados_integrator = AcadosSimSolver(sim_ct) #Used to mock data for testing mhe

nx = model_ct.x.size()[0]
nu = model_ct.u.size()[0]
u0 = nmp.zeros(nu)

#Storage structs
simX = nmp.ndarray((N+1, nx))
simU = nmp.ndarray((N, nu))
simY = nmp.ndarray((N+1, 1))

#inital state/parameters
x0 = nmp.zeros(nx)

########################
# ode used to generate data to test.
########################
simX[0,:] = x0
v_stds = [0.05]


Tstep = 0.8
simU[round(0.8/dt):, 0] = 1

for i in range(N):
    # set initial state
    u0 = simU[i,:]
    p = nmp.array([zeta, ts, Kp])
    acados_integrator.set("p", p)
    acados_integrator.set("x", x0)
    acados_integrator.set("u", u0)
    # solve
    status = acados_integrator.solve()
    # get solution
    x0 = acados_integrator.get("x")

    simX[i+1,:]  = x0
    # C = [1; 0] Measurement model
    simY[i+1, :] = x0[0] + nmp.transpose(nmp.diag(v_stds) @ nmp.random.standard_normal((1, 1)))
    if status != 0:
        raise Exception('acados returned status {}. Exiting.'.format(status))





plt.figure(1)
plt.plot(np.linspace(0, Tf, N+1), simX[:, 0], label="true")
plt.plot(np.linspace(0, Tf, N+1), simY[:, 0], 'x', label="measured")

plt.ylabel("[-]")
plt.xlabel("time [s]")
plt.title("SO ODE Dynamic Model")
plt.grid()
plt.legend(loc="upper left")
plt.show(block = False)

plt.show()

