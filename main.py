#!/usr/bin/env python
from classes import EMF, Plotter
from core import boris
import numpy as np

objEMF = EMF()
objEMF.init_const(1.)
#objEMF.init_wave(1., 100., (0., 0., 1.), 0.)

m = 1.
e = 1.

x0 = np.array([0., 1., 0.])
p0 = np.array([1., 0., 1.])
tspan = (0., 1000.)
nt = 100

ans = boris(p0, x0, e, m, objEMF, tspan, nt)
objPlt = Plotter(ans[0], ans[1], ans[2], ans[3])
objPlt.space()
