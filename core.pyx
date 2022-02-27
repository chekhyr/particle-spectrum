# cython: language_level=3str

import numpy as np
import cython
cimport numpy as np
import math as mt
from classes import EMF

cdef gamma(double[3] p):
    return mt.sqrt(1. + p[0]*p[0] + p[1]*p[1] + p[2]*p[2])

cdef Lorentz(q, v, e, h):
    return q*(e + np.cross(v, h))

cdef radFrict(void):
    return -1 # not implemented

cdef boris(double[3] p0, double[3] x0, double charge, double mass, field, double[2] t_span, int Nt):
    cdef:
        int j
        double[:] time
        double dt

        double[3][:] r
        double[3][:] p
        double[3][:] v

# integration segment
    dt = (t_span[1] - t_span[0]) / Nt
    for j in range(0, Nt+1, 1):
        time[j] = t_span[0] + j * dt

# time-dependent vectors
    cdef:
        double[3] e =  field._e(time[0], x0)
        double[3] h = field._h(time[0], x0)
        double[3] p_n_minus_half
        double[3] p_minus
        double[3] tau
        double[3] p_n_plus_half

# first step
    p[:, 0] = p0
    v[:, 0] = p0[:, 1] / (mass * gamma(p0))
    r[:, 0] = x0
    p_n_plus_half = p[:, 0] + (dt / 2) * Lorentz(charge, v[:, 0], e, h)
    r[:, 1] = r[:, 0] + dt * p_n_plus_half / (mass * gamma(p_n_plus_half))

# main cycle
    for j in range(1, Nt+1, 1):
        E = field._e(time[j], r[:, j])
        H = field._h(time[j], r[:, j])

    p_n_minus_half = p_n_plus_half
    p_minus = p_n_minus_half + charge * E * (dt / 2)
    tau = charge * H / (mass * gamma(p_minus)) * (dt / 2)
    p_n_plus_half = p_minus + np.cross(p_minus + np.cross(p_minus, tau), 2 * tau / (1 + np.dot(tau, tau))) + charge * E * (dt / 2)

    p[:, j] = 0.5 * (p_n_plus_half + p_n_minus_half)
    if j != (Nt+1):
        r[:, j + 1] = r[:, j] + dt * p_n_plus_half / (mass * gamma(p_n_plus_half))
    v[:, j] = p[:, j] / (mass * gamma(p[:, j]))

    return (r, p, v, time)

