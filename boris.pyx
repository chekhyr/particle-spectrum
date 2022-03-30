# cython: language_level=3str, boundscheck=False, wraparound=False, cdivision=True
import numpy as np # to use numpy methods
from classes import Particle, EMF

cimport numpy as np # to convert numpy into c
from libc.math cimport sqrt


cdef double dot(double[:] vec1, double[:] vec2):
    cdef double res = 0
    for i in range(3):
        res += vec1[i]*vec2[i]
    return res


cdef double cross(double[:] vec1, double[:] vec2, int j):
    if j == 0:
        return vec1[1] * vec2[2] - vec1[2] * vec2[1]
    elif j == 1:
        return vec1[2] * vec2[0] - vec1[0] * vec2[2]
    elif j == 2:
        return vec1[0] * vec2[1] - vec1[1] * vec2[0]


cpdef tuple boris_routine(ptcl: Particle, field: EMF, t_span: tuple, nt: int, rad: bool):
    cdef:
        int i, j
        double scaling, dt, temp = 0

        double p_plus[3]
        double p_minus[3]
        double p_prime[3]

        double tau[3]
        double sigma[3]

        double xtmp[3]
        double vtmp[3]

        double currE[3]
        double currH[3]


# initialization
    t = np.linspace(t_span[0], t_span[1], nt)
    x = np.zeros((nt, 3), dtype=np.double)
    p = np.zeros((nt, 3), dtype=np.double)
    dt = t[1] - t[0]

    x[0, :] = ptcl.x0
    p[0, :] = ptcl.p0

# main cycle
    for i in range(1, nt, 1):
        for j in range(3):
            xtmp[j] = x[i-1, j]
            currE[j] = field.e(xtmp, t[i])[j]
            currH[j] = field.h(xtmp, t[i])[j]

        temp = p[i-1, :].dot(p[i-1, :])
        for j in range(3):
            tau[j] = currH[j] * dt / 2 / sqrt(1 + temp)
        for j in range(3):
            sigma[j] = 2 * tau[j] / (1 + dot(tau, tau))
        for j in range(3):
            p_minus[j] = p[i-1, j] + currE[j] * dt / 2
        for j in range(3):
            p_prime[j] = p_minus[j] + cross(p_minus, tau, j)
        for j in range(3):
            p_plus[j] = p_minus[j] + cross(p_prime, sigma, j)
        for j in range(3):
            p[i, j] = p_plus[j] + currE[j] * dt / 2

        for j in range(3):
            vtmp[j] = (p[i, j] + p[i-1, j]) / 2 / sqrt(1 + temp)
        for j in range(3):
            sigma[j] = currE[j] + cross(vtmp, currH, j)  # Lorentz force
            scaling = field.omg * ptcl.q**2 / ptcl.m
            K = (1 + temp) * 0.0118 * scaling * (dot(sigma, sigma) - (dot(vtmp, sigma)) ** 2)  # letter K
        if rad == 0:
            K = 0
        for j in range(3):
            p[i, j] = p[i, j] - dt * K * vtmp[j]
        for j in range(3):
            x[i, j] = x[i-1, j] + p[i, j] * dt / sqrt(1 + temp)

    return t, x, p

#TODO: Limit interaction time by introducing external field envelope
