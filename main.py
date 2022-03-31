#!/usr/bin/env python
from boris import boris_routine, EMF
from intensity import intensity_integral
from classes import Plots, Particle, PlotIntensity
import numpy as np

# Config
nt = 10000
t_span = (0, 600)

q = 1.
m = 1.
x0 = np.array([0, 0, 0]).astype(np.double)
p0 = np.array([0.1, 0, 0]).astype(np.double)

Radiation = True


objPtcl = Particle(q, m, x0, p0)
objEMF = EMF(b'const')

trj = boris_routine(objPtcl, objEMF, t_span, nt, Radiation)

objPlt = Plots(trj[0], trj[1])
objPlt.space()

'''
nOmg = 10
intns = intensity_integral(trj[0], trj[1], trj[2], nOmg)
objIntns = PlotIntensity(intns[0], intns[1])
objIntns.draw()
'''