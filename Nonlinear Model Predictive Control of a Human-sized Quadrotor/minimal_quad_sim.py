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
from export_quad_ode_model import export_quad_ode_model
from utils import *
import numpy as np
import matplotlib.pyplot as plt

sim = AcadosSim()

# export model 
model = export_quad_ode_model()

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
# set model_name 
sim.model = model
sim.parameter_values = p
Tf = 2.5
dt = 0.01
nx = model.x.size()[0]
nu = model.u.size()[0]

print("StateSize:" ,nx)
print("ControlSize:" ,nu)

# set simulation time
sim.solver_options.T = dt
# set options
sim.solver_options.num_stages = 4
sim.solver_options.num_steps = 3
sim.solver_options.newton_iter = 3 # for implicit integrator


# create
acados_integrator = AcadosSimSolver(sim)


Nsim = int(Tf/dt)
uSS = 39.99
simX = np.ndarray((Nsim+1, nx))
simU = np.ndarray((Nsim, nu))

x0 = np.zeros((nx))
x0[0:4] = YPRtoQuat(np.array([0, 0, 0]))

u0 = np.zeros((nu))

simX[0,:] = x0


acados_integrator.set("p", p)

for i in range(Nsim):
    # set initial state
    acados_integrator.set("x", x0)
    # set control inputs
    # u0 = 0.25 * uMax * np.random.randn(4)
    #u0 = np.array([10,0,10,0]) #Positive yaw.
    #u0 = np.array([0,10,0,10])  # Negative yaw.

    u0 = np.array([uSS, uSS + np.sqrt(10 * 10 * 0.5), uSS, uSS + np.sqrt(10 * 10 * 0.5)])  # Positive yaw.
    u0 = np.array([uSS-5, uSS, uSS+5, uSS])  # Negatice yaw.

    # u0 = np.array([uSS, uSS + np.sqrt(10 * 10 * 0.5), uSS, uSS+np.sqrt(10 * 10 * 0.5)]) #Positive pitch.
    u0 = np.array([uSS+0, uSS+np.sqrt(10 * 10 * 0.5), uSS+10, uSS+np.sqrt(10 * 10 * 0.5)])  # Negative pitch.

    # u0 = np.array([np.sqrt(10*10*0.5), 10, np.sqrt(10*10*0.5), 0]) #Positive roll.
    # u0 = np.array([np.sqrt(10 * 10 * 0.5), 0, np.sqrt(10 * 10 * 0.5),10])  # Negative roll.

    acados_integrator.set("u", u0)
    # solve
    status = acados_integrator.solve()
    # get solution
    x0 = acados_integrator.get("x")

    simU[i,:] = u0
    simX[i+1,:] = x0

    if status != 0:
        raise Exception('acados returned status {}. Exiting.'.format(status))


q = np.zeros((2, 4))
q[0, 0] = 0.479
q[0, 1] = 0
q[0, 2] = 0
q[0, 3] = 0.878
q[1, 0] = 1
resEul = QuattoRPY(q)
print(resEul)

rpy = np.zeros((2, 3))
rpy[0, 0] = -np.pi
rpy[0, 1] = 0
rpy[0, 2] = 0
resQuat = YPRtoQuat(rpy)
print(resQuat)
# plot results
uDel = 8
plot_quad(dt, uSS, uDel, simU, simX)


