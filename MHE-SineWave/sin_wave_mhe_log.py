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

LOGS_Dir = "/Documents/logs"
DATA_Dir = "/287_27/"
ATT_File = "pose_body.attitudeOutput.csv"
FRAME_File = "pose_body.frameOutput.csv"
FILE_Name = LOGS_Dir + DATA_Dir

print('Data Location:', FILE_Name)

with open(FILE_Name + ATT_File, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    att_headers = next(reader)
    att_data = np.array(list(reader)).astype(float)

with open(FILE_Name + FRAME_File, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    frm_headers = next(reader)
    frm_data = np.array(list(reader)).astype(float)

print("Body Attitude:", att_data.shape[0])
print("Body Frame:", frm_data.shape)

pitchIdx = 2
timeIdx  = 0
# Plot the data
plt.figure(1)
plt.plot((att_data[:, timeIdx]-att_data[0, timeIdx])*1e-9, att_data[:, pitchIdx]*180.0/nmp.pi)
plt.xlabel(att_headers[0])
plt.ylabel(att_headers[pitchIdx] + "[deg]")
plt.title("Pitch over time")
plt.grid()
plt.show(block = False)

# log_dt = nmp.floor(nmp.mean(nmp.diff((att_data[:, timeIdx]-att_data[0, timeIdx])*1e-9))*100)/100
dt = nmp.floor(nmp.mean(nmp.diff((att_data[:, timeIdx]-att_data[0, timeIdx])*1e-9))*100)/100
Tv = (att_data[:, timeIdx]-att_data[0, timeIdx])*1e-9

#######################################################################################################
# mhe model and solver
#######################################################################################################
Tf = 4   # [4sec window]
# dt = 1/20.0
N = int(Tf/dt) #400  # 4sec at 20Hz
h = dt

# Tv20Hz = np.arange(0,Tv[-1],dt)
# att_data_20Hz = numpy.interp(Tv20Hz, Tv, att_data[:, pitchIdx])


model_mhe = export_sine_wave_mhe_ode_model()

nx_augmented = model_mhe.x.size()[0]
nu = model_mhe.u.size()[0]
nw = model_mhe.u.size()[0]
nx = nx_augmented - 4

x0 = nmp.zeros((nx_augmented))
x0 = nmp.array([0, 0.1, 1, 1, 0])

u0 = nmp.zeros((nu))

Nloops = att_data.shape[0] - (N+1)
mheXest = nmp.zeros((Nloops, N+1, nx_augmented))
mheXest[0, 0, :] = x0
mheWest = nmp.zeros((Nloops, N,   1))
mhePest = nmp.zeros((Nloops, N+1, 4))
mheTe   = nmp.zeros((Nloops, N+1, 1)) # Time vector for the estimation horizon
mheYref = nmp.zeros((Nloops, N,   1))
# Predict forward 500ms
tp = 0.5
Nfwd = int(tp/dt)
mheYpred = nmp.zeros((Nloops, Nfwd, nx))
mheYtrue = nmp.zeros((Nloops, Nfwd, nx))
mheTp = nmp.zeros((Nloops, Nfwd, 1))  # Time vector for the predictions horizon
Q0_mhe = np.diag([0.1, 0.1, 0.1, 0.1, 0.1])
Q_mhe  = 10.*np.diag([0.1])
R_mhe  = 200.*np.diag([0.1])

acados_solver_mhe = export_sine_wave_mhe_solver(model_mhe, N, h, Q_mhe, Q0_mhe, R_mhe)

print("Nfwd:", Nfwd)
#######################################################################################################
# prediction/forward integration model
#######################################################################################################
sim = AcadosSim()
# export model
sim_model = export_sine_wave_mhe_ode_model()
sim.model = sim_model
sim.parameter_values = np.array([0])
# set simulation time
sim.solver_options.T = dt
# set options
sim.solver_options.integrator_type = 'ERK'
sim.solver_options.num_stages = 4
sim.solver_options.num_steps = 3
sim.solver_options.newton_iter = 3 # for implicit integrator
# create
acados_integrator = AcadosSimSolver(sim) #Used to mock data for testing mhe

#TODO:
# loop through logged data, set initial state and parameters then store estimation horizon data/estimate.
# set measurements and controls
skipSeconds = 25
Nskip =  int(skipSeconds/dt)
Nmhe = 1000 #Number of mhe iterations to perform
for i in range(Nskip, Nskip+Nmhe): #att_data.shape[0]):
    # att_data[:, pitchIdx]
    tc = nmp.array([i*dt])  # Current time
    te = nmp.array([tc - N*dt])  # Estimation window termination time
    print("te:", te)
    print("tc:", tc)
    # print("tp:", tc)
    yref_0 = np.zeros((2*nx + nx_augmented,))
    yref_0[:nx] = att_data[i-N, pitchIdx]

    yref_0[2*nx:] = x0
    acados_solver_mhe.set(0, "yref", yref_0)
    acados_solver_mhe.set(0, "p", te)

    # set initial guess to x0
    acados_solver_mhe.set(0, "x", x0)

    yref = np.zeros((2*nx, ))
    mheYref[i, 0, 0] = att_data[i - N, pitchIdx]
    mheTe[i, 0, 0] = nmp.array([te])
    for j in range(1, N):
        # set measurements and controls
        yref[:nx] = att_data[(i-N) + j, pitchIdx]
        acados_solver_mhe.set(j, "yref", yref)
        acados_solver_mhe.set(j, "p", nmp.array([te+j*dt]))
        mheYref[i, j, 0] = att_data[(i-N) + j, pitchIdx]
        mheTe[i, j, 0] = nmp.array([te+j*dt])
        # set initial guess to x0_bar
        acados_solver_mhe.set(j, "x", x0)

    mheTe[i, N, 0] = nmp.array([te + N*dt])
    acados_solver_mhe.set(N, "x", x0)
    # solve mhe problem
    status = acados_solver_mhe.solve()
    if status != 0:
        raise Exception('acados returned status {}. Exiting.'.format(status))

    # get estimation solution
    for j in range(N):
        x_augmented = acados_solver_mhe.get(j, "x")
        mheXest[i, j, :] = x_augmented[0:nx]
        mhePest[i, j, :] = x_augmented[nx]
        mheWest[i, j, :] = acados_solver_mhe.get(j, "u")

    x_augmented = acados_solver_mhe.get(N, "x")
    mheXest[i, N, :] = x_augmented[0:nx]
    mhePest[i, N, :] = x_augmented[nx]
    x0 = x_augmented

    # Check how well the model used in the mhe does with predicting forward.
    # get prediction solution
    pred_x0 = x0
    pred_u0 = nmp.zeros((sim_model.u.size()[0]))
    for j in range(Nfwd):
        # set initial state
        pred_p = nmp.array([tc + j*dt])
        acados_integrator.set("p", pred_p)
        acados_integrator.set("x", pred_x0)
        acados_integrator.set("u", pred_u0)
        # solve
        status = acados_integrator.solve()
        # get solution
        pred_x0 = acados_integrator.get("x")
        mheTp[i, j, 0] = nmp.array([i*dt + j*dt])
        if status != 0:
            raise Exception('acados returned status {}. Exiting.'.format(status))
        mheYpred[i, j, 0] = pred_x0[0]
        mheYtrue[i, j, 0] = att_data[i + j, pitchIdx]
        # print("tP: %.2f \t  yP: %.4f \t yT: %.4f" % (mheTp[i, j, 0],  mheYpred[i, j, 0], mheYtrue[i, j, 0]))
        # print("tP:",, "yP:",, "yT:", )



from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots()
ax.grid(True)

#Estimations lines
mhe_fit = ax.plot(mheTe[Nskip, :, 0], mheXest[Nskip, :, 0], color='b', lw=2)[0]
mhe_true = ax.plot(mheTe[Nskip, :-1, 0], mheYref[Nskip, :, 0], color='k', lw=2)[0]
mhe_fit.set_label('mhe fit')
mhe_true.set_label('mhe yref')
#Prediction lines
mpc_fit = ax.plot(mheTp[Nskip, :, 0], mheYpred[Nskip, :, 0], color='r', lw=2)[0]
mpc_true = ax.plot(mheTp[Nskip, :, 0], mheYtrue[Nskip, :, 0], color='k',linestyle='--', lw=2)[0]
mpc_fit.set_label('mpc fit')
mpc_true.set_label('mpc yref')
def animate(i):
    #TODO Set the time limits for the estimation horizon.
    # Plot estimation vs true data
    # Set the time limits for the prediction horizon
    # Plot the prediction vs true data
    # mhe_fit.set_xdata(mheTe[Nskip + i, :, 0])
    # mhe_fit.set_ydata(mheXest[Nskip + i, :, 0])
    mhe_fit.set_xdata(mheTe[Nskip + i, :, 0])
    mhe_fit.set_ydata(mheXest[Nskip + i, :, 0])
    mhe_true.set_xdata(mheTe[Nskip + i, :-1, 0])
    mhe_true.set_ydata(mheYref[Nskip + i, :, 0])
    #
    mpc_fit.set_xdata(mheTp[Nskip + i, :, 0])
    mpc_fit.set_ydata(mheYpred[Nskip + i, :, 0])
    mpc_true.set_xdata(mheTp[Nskip + i, :, 0])
    mpc_true.set_ydata(mheYtrue[Nskip + i, :, 0])

    ax.relim()
    ax.autoscale_view(True, True, True)
    return [mhe_fit,mhe_true,mpc_fit,mpc_true]

anim = FuncAnimation(
    fig, animate, interval=100, frames=Nmhe-1)


plt.show()

