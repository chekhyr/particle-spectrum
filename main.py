#!/usr/bin/env python
from boris import boris_routine, EMF, Particle
from spectrum import spectrum_routine
from plotter import PlotTrajectory, PlotSpectrum
import numpy as np

# Config
nt = 1000
t_span = (0., 200.)

q = 1.
m = 1.
x0 = np.array([0, 0, 0]).astype(np.double)
p0 = np.array([0.9, 0, 0]).astype(np.double)

Radiation = True


objPtcl = Particle(q, m, x0, p0)
objEMF = EMF(0)

trj = boris_routine(objPtcl, objEMF, t_span, nt, Radiation)

objPlt = PlotTrajectory(trj[0], trj[1])
objPlt.space()

n = np.array([0, 1, 0]).astype(np.double)
omg_span = (0., 4.)
nOmg = 500
spct = spectrum_routine(trj[0], trj[1], trj[2], trj[3], n, omg_span, nOmg, objPtcl)

objSpct = PlotSpectrum(spct[0], spct[1])
objSpct.draw()
