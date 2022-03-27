import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt


class EMF:
    __slots__ = ['e', 'h']

    def __init__(self):
        self.e = None
        self.h = None

    def init_const(self):
        ampl = np.double(1)
        def e(x, t):
            return ampl * np.array([0, 0, 0]).astype(np.double)

        def h(x, t):
            return ampl * np.array([0, 0, 1]).astype(np.double)

        self.e = e
        self.h = h

    def init_wave(self):
        ampl = np.double(1)
        omg = np.double(1)
        k = np.array([0., 0., 1.]).astype(np.double)
        alph = np.double(0)

        k = k / np.sqrt(k.dot(k))

        def e(x, t):
            ex = ampl * math.cos(omg * t - np.dot(k, x) + alph)
            ey = ampl * math.sin(omg * t - np.dot(k, x) + alph)
            ez = 0
            return np.array([ex, ey, ez]).astype(np.double)

        def h(x, t):
            return np.cross(e(x, t), _k)

        self.e = e
        self.h = h

    def init_gauss(self):
        print('err: Gaussian beam not implemented')
        return -1



