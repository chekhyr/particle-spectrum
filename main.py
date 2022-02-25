#!/usr/bin/env python
from core import EMF
import numpy as np


obj = EMF('debug', [1., 20., 1.])
x = np.array([0., 0., 6.])
print(obj(x, 8.))
