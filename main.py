#!/usr/bin/env python
from classes import EMF
import numpy as np

obj = EMF()
obj.init_wave()

x = np.array([0., 0., 10.])
t = 8.
print(obj(x, t))
