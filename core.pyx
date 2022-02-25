# cython: language_level=3str

import numpy as np
import cython
cimport numpy as np
import math


cdef class EMF:
    cdef:
            double ampl
            double omg
            double alph
            double[:] k
            str mode

    def __init__(self, str mode, list k=[0., 0., 0.], double ampl=1., double omg=1., double alph=0.):
        self.mode = mode
        _k = np.array(k)
        self.k = _k/np.sqrt(_k.dot(_k)) # propagation direction
        self.alph = alph  # initial phase (field orientation)
        self.omg = omg  # oscillation frequency
        self.ampl = ampl

    def __call__(self, x, t):
        if self.mode == 'magnet':
            e = self.ampl*np.array([0., 0., 0.])
            h = self.ampl*np.array([0., 0., 1.])
            return (e, h)
        elif self.mode == 'monowave':
            ex = self.ampl*math.cos(self.omg*t - np.dot(self.k, x) + self.alph)
            ey = self.ampl*math.sin(self.omg*t - np.dot(self.k, x) + self.alph)
            ez = 0
            e = np.array([ex, ey, ez])
            h = np.cross(e, self.k)
            return (e, h)
        elif self.mode == 'gauss':
            print('err: Gaussian beam not implemented')
            return -1
        elif self.mode == 'debug':
            print('why?')
            return 0

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
