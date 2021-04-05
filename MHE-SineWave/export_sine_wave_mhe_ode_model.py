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

from acados_template import AcadosModel
from casadi import *
import numpy as np
def export_sine_wave_mhe_ode_model():

    model_name = 'sine_wave_ode'

    # set up states & controls
    s     = SX.sym('s')
    amp   = SX.sym('amp')
    freq  = SX.sym('freq')
    phase = SX.sym('phase')
    trend = SX.sym('trend')
    x = vertcat(s, amp, freq, phase, trend)

    # (controls) state noise
    w_s     = SX.sym('w_s')

    w = vertcat(w_s)
    
    # xdot
    s_dot     = SX.sym('s_dot')
    amp_dot   = SX.sym('amp_dot')
    freq_dot  = SX.sym('freq_dot')
    phase_dot = SX.sym('phase_dot')
    trend_dot = SX.sym('trend_dot')

    xdot = vertcat(s_dot, amp_dot, freq_dot, phase_dot, trend_dot)

    # algebraic variables
    # z = None

    # parameters
    # set up parameters
    t  = SX.sym('t')   # time
    p = vertcat(t)

    # dynamics
    f_expl = vertcat(
                    trend + amp*cos(2*pi*freq*t + phase)*2*pi*freq,
                    0,
                    0,
                    0,
                    0
                     )
    f_expl[0] = f_expl[0] + w
    f_impl = xdot - f_expl
    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = w
    # model.z = []
    model.p = p
    model.name = model_name

    return model

