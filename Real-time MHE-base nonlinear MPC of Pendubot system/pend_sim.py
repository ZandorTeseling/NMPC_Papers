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
from export_pend_ode_model import export_pend_ode_model
from utils import *
import numpy as nmp
import matplotlib.pyplot as plt

sim = AcadosSim()

# export model 
model = export_pend_ode_model()

# set model_name 
sim.model = model

Tf = 10.0
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
uMax = 2 #N/m control input max
simX = nmp.ndarray((Nsim+1, nx))
simU = nmp.ndarray((Nsim, nu))

x0 = nmp.zeros((nx))
u0 = nmp.zeros((nu))

#Parameters
# set up parameters
m1 = 0.265  # mass link 1
m2 = 0.226  # mass link 2
l1 = 0.206  # length link 1
l2 = 0.298  # length link 2
lc1 = 0.107  # cm link 1
lc2 = 0.133  # cm link 2
b1 = 0.35  # damping coef joint 1
b2 = 0.45  # damping coef joint 2
g = 9.81

x0[4] = b1
x0[5] = b2

simX[0,:] = x0

p = nmp.array([m1, m2, l1, l2, lc1, lc2, g])
acados_integrator.set("p", p)

for i in range(Nsim):
    # set initial state
    acados_integrator.set("x", x0)
    # set control inputs
    sigma = 1
    # u0 = nmp.random.randn()*sigma
    acados_integrator.set("u", u0)
    # solve
    status = acados_integrator.solve()
    # get solution
    x0 = acados_integrator.get("x")

    simU[i,:] = u0
    simX[i+1,:] = x0

    if status != 0:
        raise Exception('acados returned status {}. Exiting.'.format(status))


# plot results
plot_double_pendulum(dt, uMax, simU, simX)


