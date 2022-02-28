# cython: language_level=3str

import numpy as np
import cython
cimport numpy as np
import math as mt
from classes import EMF

cdef gamma(np.ndarray p):
    return mt.sqrt(1. + p.dot(p))

cdef Lorentz(double q, np.ndarray v, np.ndarray e, np.ndarray h):
    return q*(e + np.cross(v, h))

'''
cdef radFrict(void):
    return -1
''' # radiational friction is not implemented

cpdef boris(np.ndarray p0, np.ndarray x0, double charge, double mass, field, tuple t_span, int Nt):
    cdef:
        int j
        double dt

        np.ndarray time
        np.ndarray r
        np.ndarray p
        np.ndarray v

# integration segment
    time = np.linspace(t_span[0], t_span[1], Nt)
    dt = time[1] - time[0]
    r = np.zeros((3, Nt))
    p = np.zeros((3, Nt))
    v = np.zeros((3, Nt))

# time-dependent vectors
    cdef:
        np.ndarray e =  field._e(time[0], x0)
        np.ndarray h = field._h(time[0], x0)
        np.ndarray p_n_minus_half
        np.ndarray p_minus = np.zeros(3)
        np.ndarray tau
        np.ndarray p_n_plus_half

# first step
    p[:, 0] = p0
    v[:, 0] = np.divide(p0, (mass * gamma(p0)))
    r[:, 0] = x0
    p_n_plus_half = p[:, 0] + (dt / 2) * Lorentz(charge, v[:, 0], e, h)
    r[:, 1] = r[:, 0] + dt * np.divide(p_n_plus_half, (mass * gamma(p_n_plus_half)))

# main cycle
    for j in range(1, Nt, 1):
        E = field._e(time[j], r[:, j])
        H = field._h(time[j], r[:, j])

    p_n_minus_half = p_n_plus_half
    p_minus[:] = p_n_minus_half[:] + charge * E * (dt / 2)
    tau = charge * H / (mass * gamma(p_minus)) * (dt / 2)
    p_n_plus_half = p_minus + np.cross(p_minus + np.cross(p_minus, tau), 2 * np.divide(tau, (1 + np.dot(tau, tau)))) + charge * E * (dt / 2)

    p[:, j] = 0.5 * (p_n_plus_half + p_n_minus_half)
    if j != Nt-1:
        r[:, j + 1] = r[:, j] + dt * np.divide(p_n_plus_half, (mass * gamma(p_n_plus_half)))
    v[:, j] = p[:, j] / (mass * gamma(p[:, j]))

    return (r, p, v, time)

