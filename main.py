#!/usr/bin/env python
from classes import EMF
from core import boris
import numpy as np
import matplotlib.pyplot as plt

obj = EMF()
obj.init_const()

x0 = np.array([0., 0., 1.])
p0 = np.array([1., 1., 1.])
tspan = (0., 10.)
t = 8.
print(boris(p0, x0, 1., 1., obj, tspan, 100))
