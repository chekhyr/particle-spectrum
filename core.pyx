# cython: language_level=3str

import numpy as np
import cython
cimport numpy as np
import math as mt
from classes import EMF

cdef gamma(p: np.ndarray[3]):
    return mt.sqrt(1. + p.dot(p))

cdef lorentz(double q, np.ndarray v, np.ndarray e, np.ndarray h):
    return q*(e + np.cross(v, h))

#TODO: Introduce reaction force and modify boris() accordingly
'''
cdef radFrict(void):
    return -1
''' # radiation reaction force is not implemented

cpdef boris(p0: np.ndarray, x0: np.ndarray, charge: float, mass: float, field: EMF, t_span: tuple, nt: int):
    cdef:
    # integration segment
        np.ndarray time = np.linspace(t_span[0], t_span[1], nt)
        double dt = time[1] - time[0]
    # answer
        np.ndarray r = np.zeros((nt, 3))
        np.ndarray p = np.zeros((nt, 3))
        np.ndarray v = np.zeros((nt, 3))

# time-dependent vectors
    cdef:
        np.ndarray e = field.e(x0, time[0])
        np.ndarray h = field.h(x0, time[0])

        np.ndarray p_n_plus_half
        np.ndarray p_n_minus_half
        np.ndarray p_minus
        np.ndarray tau

# first step
    p[0, :] = p0
    v[0, :] = np.divide(p0, (mass * gamma(p0)))
    r[0, :] = x0
    p_n_plus_half = p0 + (dt / 2) * lorentz(charge, v[0, :], e, h)
    r[1, :] = r[0, :] + dt * np.divide(p_n_plus_half, (mass * gamma(p_n_plus_half)))

# main cycle
    for j in range(1, nt-1, 1):
        e = field.e(r[j, :], time[j])
        h = field.h(r[j, :], time[j])

        p_n_minus_half = p_n_plus_half
        p_minus = np.add(p_n_minus_half, np.multiply(e, charge) * (dt / 2))
        tau = np.divide(np.multiply(h, charge), (mass * gamma(p_minus)) * (dt / 2))
        p_n_plus_half = p_minus + np.cross(p_minus + np.cross(p_minus, tau), 2 * np.divide(tau, (1 + np.dot(tau, tau)))) + np.multiply(e, charge) * (dt / 2)

        p[j, :] = 0.5 * (p_n_plus_half + p_n_minus_half)
        if j != nt-1:
            r[j+1, :] = r[j, :] + dt * np.divide(p_n_plus_half, (mass * gamma(p_n_plus_half)))
        v[j, :] = np.divide(p[j, :], (mass * gamma(p[j, :])))

    return r, p, v, time

#TODO: Limit interaction time by introducing external field envelope

#TODO: Add function to find integrated emission spectrum into a given angle

#TODO: Add function to draw spectrum colormap for a given angle interval (all angles)
