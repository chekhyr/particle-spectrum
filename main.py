#!/usr/bin/env python
from boris import boris_routine
from classes import Plots, EMF, Particle
import numpy as np

# Config
nt = 10000
t_span = (0, 600)

q = 1.
m = 1.
x0 = np.array([0, 0, 0]).astype(np.double)
p0 = np.array([0.1, 0, 1]).astype(np.double)

Radiation = 1


objPtcl = Particle(q, m, x0, p0)
objEMF = EMF()
#objEMF.init_const()
objEMF.init_wave()

trj = boris_routine(objPtcl, objEMF, t_span, nt, Radiation)

objPlt = Plots(trj[1], trj[0])
objPlt.space()