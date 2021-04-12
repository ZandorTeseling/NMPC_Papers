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
from export_sine_wave_mhe_ode_model import export_sine_wave_mhe_ode_model
from export_sine_wave_mhe_solver import export_sine_wave_mhe_solver
from utils import *
from casadi import *
import numpy as nmp
import csv
import matplotlib.pyplot as plt

sinFunction = lambda x, tVec : x[0] + x[1]*sin(2*pi*x[2]*tVec + x[3]) + x[4]*tVec
sim = AcadosSim()
# export model
model = export_sine_wave_mhe_ode_model()

dt = 0.01
Tf = 2.0
N = int(Tf/dt)
sim.model = model
t = 0

sim.parameter_values = np.array([t])
# set simulation time
sim.solver_options.T = dt
# set options
sim.solver_options.integrator_type = 'ERK'
sim.solver_options.num_stages = 4
sim.solver_options.num_steps = 3
sim.solver_options.newton_iter = 3 # for implicit integrator
# create
acados_integrator = AcadosSimSolver(sim) #Used to mock data for testing mhe

nx_augmented = model.x.size()[0]
nu = model.u.size()[0]
nx = nx_augmented - 4
x0 = nmp.zeros((nx_augmented))
u0 = nmp.zeros((nu))

#Storage structs
simX = nmp.ndarray((N+1, nx_augmented))
simU = nmp.ndarray((N, nu))
simY = nmp.ndarray((N+1, 1))

simXest = nmp.zeros((N+1, nx_augmented))
simWest = nmp.zeros((N, 1))
simPest = nmp.zeros((N+1, 4))

#inital state/parameters
x0 = nmp.array([-1, 1, 4.4, 0, 1]) #s, amp, freq, phase, trend

########################
# ode used to generate data to test.
########################
simX[0,:] = x0
v_stds = [0.4]

for i in range(N):
    # set initial state
    p = nmp.array([i*dt])
    acados_integrator.set("p", p)
    acados_integrator.set("x", x0)
    acados_integrator.set("u", u0)
    # solve
    status = acados_integrator.solve()
    # get solution
    x0 = acados_integrator.get("x")

    simX[i+1,:]  = x0
    simY[i+1, :] = x0[0] + nmp.transpose(nmp.diag(v_stds) @ nmp.random.standard_normal((nx, 1)))
    if status != 0:
        raise Exception('acados returned status {}. Exiting.'.format(status))

########################
# mhe model and solver
########################
h = Tf/N #dt
model_mhe = export_sine_wave_mhe_ode_model()

nw = model_mhe.u.size()[0]
ny = nx
Q0_mhe = np.diag([0.1, 0.1, 0.1, 0.1, 0.1])
Q_mhe  = 10.*np.diag([0.1])
R_mhe  = 2*np.diag([0.1])

acados_solver_mhe = export_sine_wave_mhe_solver(model_mhe, N, h, Q_mhe, Q0_mhe, R_mhe)

# set measurements and controls
x0_bar = np.array([4.1, 0.8, 0.8, 0.5, 0.8])

yref_0 = np.zeros((2*nx + nx_augmented, ))
yref_0[:nx] = simY[0, :]
yref_0[2*nx:] = x0_bar

acados_solver_mhe.set(0, "yref", yref_0)
acados_solver_mhe.set(0, "p", simU[0,:])

# set initial guess to x0_bar
acados_solver_mhe.set(0, "x", x0_bar)

yref = np.zeros((2*nx, ))
for j in range(1, N):
    # set measurements and controls
    yref[:nx] = simY[j, :]
    acados_solver_mhe.set(j, "yref", yref)
    acados_solver_mhe.set(j, "p", nmp.array([j*dt]))

    # set initial guess to x0_bar
    acados_solver_mhe.set(j, "x", x0)

acados_solver_mhe.set(N, "x", x0)

# solve mhe problem
status = acados_solver_mhe.solve()

if status != 0:
    raise Exception('acados returned status {}. Exiting.'.format(status))

# get solution
for i in range(N):
    x_augmented = acados_solver_mhe.get(i, "x")
    simXest[i, :] = x_augmented[0:nx]
    simPest[i, :] = x_augmented[nx:]
    simWest[i, :] = acados_solver_mhe.get(i, "u")

x_augmented = acados_solver_mhe.get(N, "x")
simXest[N, :] = x_augmented[0:nx]
simPest[N, :] = x_augmented[nx]

print(x_augmented)
print('difference |x_est - x_true|', np.linalg.norm(simXest - simX))
print('difference |y_est - y_true|', np.linalg.norm(simXest - simX))

plt.figure(2)
plt.plot(np.linspace(0, Tf, N+1), simX[:, 0], label="true")
plt.plot(np.linspace(0, Tf, N+1), simY[:, 0], 'x', label="measured")
plt.plot(np.linspace(0, Tf, N+1), simXest[:, 0], label="est")
plt.ylabel("[rad]")
plt.title("Sin Dynamic Model")
plt.grid()
plt.legend(loc="upper left")
plt.show(block = False)

plt.figure(3)
plt.step(np.linspace(0, Tf, N+1), simPest[:, 0], label="amp")
plt.step(np.linspace(0, Tf, N+1), simPest[:, 1], label="freq")
plt.step(np.linspace(0, Tf, N+1), simPest[:, 2], label="phase")
plt.step(np.linspace(0, Tf, N+1), simPest[:, 3], label="trend")
plt.step(np.linspace(0, Tf, N), simWest[:, 0], label="noise w")
plt.ylabel("[states/params]")
plt.title("MHE Estimates")
plt.grid()
plt.legend(loc="upper left")
plt.show(block = False)

plt.show()

