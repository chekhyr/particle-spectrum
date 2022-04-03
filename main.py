#!/usr/bin/env python
from boris import boris_routine, EMF, Particle
from spectrum import spectrum_routine
from plotter import PlotTrajectory, PlotSpectrum
import numpy as np

import matplotlib.pyplot as plt

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
#objPlt.space()


n = np.array([0, 1, 0]).astype(np.double)
omg_span = (0., 4.)
nOmg = 500
spct = spectrum_routine(trj[0], trj[1], trj[2], trj[3], n, omg_span, nOmg, objPtcl)

objSpct = PlotSpectrum(spct[0], spct[1])
#objSpct.draw()


tht_span = (0., np.pi)
nTht = 100

tht = np.linspace(tht_span[0], tht_span[1], nTht)
heatmp = np.zeros((nTht, nOmg))
for i in range(nTht):
    n[0] = np.sin(tht[i])
    n[1] = 0
    n[2] = np.cos(tht[i])
    heatmp[i][:] = spectrum_routine(trj[0], trj[1], trj[2], trj[3], n, omg_span, nOmg, objPtcl)[1][:]
else:
    omg = np.linspace(omg_span[0], omg_span[1], nOmg)

X, Y = np.meshgrid(omg, tht)

plt.subplot(projection="polar")
plt.grid(False)
plt.pcolormesh(Y, X, heatmp)
#plt.grid()
plt.colorbar()
plt.tight_layout()
plt.show()