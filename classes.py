import numpy as np
import math

class EMF:
    __slots__ = ['e', 'h']
    def __init__(self):
        self.e = None
        self.h = None

    def init_const(self):
        def e(x, t):
            return np.array([0., 0., 0.]).astype(np.double)
        def h(x, t):
            return np.array([0., 0., 1.]).astype(np.double)
        self.e = e
        self.h = h

    def init_wave(self, k=[0., 0., 1.], ampl=1., omg=1., alph=0.):
        _k = np.array(k).astype(np.double)
        _k = _k / np.sqrt(_k.dot(_k))
        def e(x, t):
            ex = ampl * math.cos(omg * t - np.dot(_k, x) + alph)
            ey = ampl * math.sin(omg * t - np.dot(_k, x) + alph)
            ez = 0
            return np.array([ex, ey, ez]).astype(np.double)
        def h(x, t):
            return np.cross(e(x, t), _k)
        self.e = e
        self.h = h

    def init_gauss(self, ):
        print('err: Gaussian beam not implemented')
        return -1

    """
    def __call__(self, x, t):
        return (self._e(x, t), self._h(x, t))
    """

class Particle:
    def __init__(self, x, p, field):
        self.x = np.array(x).astype(np.double)
        self.p = np.array(p).astype(np.double)
        self.field = field