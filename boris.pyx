# cython: language_level=3str, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np # to use numpy methods

cimport numpy as np # to convert numpy into c
from libc.math cimport sqrt, sin, cos


cdef class Particle:
    def __init__(self, double q, double m, double[:] x0, double[:] p0):
        self.q = q
        self.m = m
        self.x0 = x0
        self.p0 = p0


cdef double dot(double[:] vec1, double[:] vec2) nogil:
    cdef double res = 0
    for i in range(3):
        res += vec1[i]*vec2[i]
    return res


cdef double cross(double[:] vec1, double[:] vec2, int k) nogil:
    if k == 0:
        return vec1[1] * vec2[2] - vec1[2] * vec2[1]
    elif k == 1:
        return vec1[2] * vec2[0] - vec1[0] * vec2[2]
    elif k == 2:
        return vec1[0] * vec2[1] - vec1[1] * vec2[0]


cdef class EMF:
    def __init__(self, int par):
        self.par = par
        self.k = (0, 0, 1)
        self.k_mv = self.k
        if self.par == 0:
            self.omg = 0
        elif self.par == 1:
            self.alph = 0
            self.omg = sqrt(dot(self.k_mv, self.k_mv))
            for i in range(3):
                self.k[i] /= self.omg


    cdef double e(self, double[:] x, double t, int k) nogil:
        cdef double res

        if self.par == 0:
            res = 0
        elif self.par == 1:
            if k == 0:
                res = cos(self.omg * t - dot(self.k_mv, x) + self.alph)
            elif k == 1:
                res = sin(self.omg * t - dot(self.k_mv, x) + self.alph)
            elif k == 2:
                res = 0
        elif self.par == 3:
            raise NotImplementedError

        return res

    cdef double h(self, double[:] x, double t, int k) nogil:
        cdef double res

        if self.par == 0:
            if k == 0 or k == 1:
                res = 0
            elif k == 2:
                res = 1
        elif self.par == 1:
            if k == 0:
                res = -self.k_mv[2]*sin(self.omg * t - dot(self.k_mv, x) + self.alph)
            elif k == 1:
                res =  self.k_mv[2]*cos(self.omg * t - dot(self.k_mv, x) + self.alph)
            elif k == 2:
                res = self.k_mv[0]*sin(self.omg * t - dot(self.k_mv, x) + self.alph) \
                      - self.k_mv[1]*cos(self.omg * t - dot(self.k_mv, x) + self.alph)
        elif self.par == 3:
            raise NotImplementedError

        return res

cpdef tuple boris_routine(ptcl: Particle, field: EMF, t_span: tuple, nt: int, rad: bool):
    cdef:
        int i, k, swtch, stp
        double scaling, K, dt, temp = 0

        double p_plus[3]
        double p_minus[3]
        double p_prime[3]

        double tau[3]
        double sigma[3]

        double xtmp[3]
        double averV[3]

        double currE[3]
        double currH[3]

        double[:] t_mv
        double[:, :] x_mv
        double[:, :] p_mv
        double[:, :] v_mv

        double[:] p_minus_mv
        double[:] p_prime_mv

        double[:] tau_mv
        double[:] sigma_mv

        double[:] averV_mv

        double[:] currE_mv
        double[:] currH_mv

# initialization
    t = np.linspace(t_span[0], t_span[1], nt)
    x = np.zeros((nt, 3), dtype=np.double)
    p = np.zeros((nt, 3), dtype=np.double)
    v = np.zeros((nt, 3), dtype=np.double)
    swtch = rad
    stp = nt

    t_mv = t
    x_mv = x
    p_mv = p
    v_mv = v

    dt = t[1] - t[0]
    scaling = field.omg * ptcl.q ** 2 / ptcl.m
    K = 0

    x[0, :] = ptcl.x0
    p[0, :] = ptcl.p0
    for k in range(3):
        v[0, k] = ptcl.p0[k] / sqrt(1+dot(p[0, :], p[0, :]))

    p_minus_mv = p_minus
    p_prime_mv = p_prime
    tau_mv = tau
    sigma_mv = sigma
    averV_mv = averV
    currE_mv = currE
    currH_mv = currH

# main cycle
    for i in range(1, stp, 1):
        for k in range(3):
            currE_mv[k] = field.e(x_mv[i-1, :], t_mv[i], k)
            currH_mv[k] = field.h(x_mv[i-1, :], t_mv[i], k)

        temp = dot(p_mv[i-1, :], p_mv[i-1, :])
        for k in range(3):
            tau[k] = currH[k] * dt / 2 / sqrt(1 + temp)
        for k in range(3):
            sigma[k] = 2 * tau[k] / (1 + dot(tau_mv, tau_mv))
        for k in range(3):
            p_minus[k] = p_mv[i-1, k] + currE[k] * dt / 2
        for k in range(3):
            p_prime[k] = p_minus[k] + cross(p_minus_mv, tau_mv, k)
        for k in range(3):
            p_plus[k] = p_minus[k] + cross(p_prime_mv, sigma_mv, k)
        for k in range(3):
            p_mv[i, k] = p_plus[k] + currE[k] * dt / 2

        for k in range(3):
            averV[k] = (p_mv[i, k] + p_mv[i-1, k]) / 2 / sqrt(1 + temp)
        for k in range(3):
            sigma[k] = currE[k] + cross(averV_mv, currH_mv, k)  # Lorentz force (here array sigma is reused)
            K = (1 + temp) * swtch * 0.0118 * scaling * (dot(sigma_mv, sigma_mv) - (dot(averV_mv, sigma_mv)) ** 2)  # letter K
        for k in range(3):
            p_mv[i, k] = p_mv[i, k] - dt * K * averV[k]
        for k in range(3):
            x_mv[i, k] = x_mv[i-1, k] + p_mv[i, k] * dt / sqrt(1 + temp)
        for k in range(3):
            v_mv[i, k] = p_mv[i, k] / sqrt(1 + temp)

    return t, x, p, v

#TODO: Limit interaction time by introducing external field envelope
