# cython: language_level=3str

import numpy as np
import cython
cimport numpy as np
import math


cdef class EMF:
    def __init__(self):
        self._e = None
        self._h = None

    def init_const(self, ampl=1.):
        def e(x, t):
            return ampl * np.array([0., 0., 0.])
        def h(x, t):
            return ampl * np.array([0., 0., 1.])
        self._e = e
        self._h = h

    def init_wave(self, list k=[0., 0., 0.], ampl=1., omg=1., alph=0.):
        _k = np.array(k)
        _k = _k / np.sqrt(_k.dot(_k))
        def e(x, t):
            ex = ampl * math.cos(omg * t - np.dot(_k, x) + alph)
            ey = ampl * math.sin(omg * t - np.dot(_k, x) + alph)
            ez = 0
            return np.array([ex, ey, ez])
        def h(x, t):
            return np.cross(e(x, t), _k)
        self._e = e
        self._h = h

    def init_gauss(self, ):
        print('err: Gaussian beam not implemented')
        return -1

    def __call__(self, x, t):
        return (self._e(x, t), self._h(x, t))


'''
cdef class Particle:
    cdef:
        double [:] x_mv, p_mv
        np.ndarray x, p
        EMF field

    def __init__(self, x, p, field):
        self.x = np.array(x).astype(np.double)
        self.p = np.array(p).astype(np.double)
        self.x_mv = self.x
        self.p_mv = self.p
        self.field = field

cdef gamma(p):
    return math.sqrt(1. + p.dot(p))

cdef Lorentz(q, v, e, h):
    return q*(e + np.cross(v, h))

cpdef boris(p0, x0, charge, mass, field, t_span, t_fragm):
    return -1
'''
