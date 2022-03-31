# cython: language_level=3str, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np # to use numpy methods
from classes import Particle
cimport numpy as np # to convert numpy into c
from libc.math cimport sqrt, sin, cos


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

cdef class EMF:
    cdef:
        char* par
        double omg
        double alph
        double k[3]

        double[:] k_mv

    def __init__(self, char* par):
        self.par = par
        self.k = (0, 0, 1)
        self.k_mv = self.k
        if self.par == b'const':
            self.omg = 0
        elif self.par == b'wave':
            self.alph = 0
            self.omg = dot(self.k_mv, self.k_mv)
            for i in range(3):
                self.k[i] /= sqrt(self.omg)


    cdef double e(self, double[:] x, double t, j):
        cdef:
            double res

        if self.par == b'const':
            res = 0

        elif self.par == b'wave':
            if j == 0:
                res = cos(self.omg * t - dot(self.k_mv, x) + self.alph)
            elif j == 1:
                res = sin(self.omg * t - dot(self.k_mv, x) + self.alph)
            elif j == 2:
                res = 0

        elif self.par == b'gauss':
            raise NotImplementedError

        return res

    cdef double h(self, double[:] x, double t, j):
        cdef:
            double res
            double tmp[3]

            double[:] tmp_mv

        for i in range(3):
            tmp[i] = self.e(x, t, i)
        tmp_mv = tmp

        if self.par == b'const':
            if j == 0 or j == 1:
                res = 0
            elif j == 2:
                res = 1

        elif self.par == b'wave':
            for i in range(3):
                res = cross(tmp_mv, self.k_mv, j)

        elif self.par == b'gauss':
            raise NotImplementedError

        return res


cpdef tuple boris_routine(ptcl: Particle, field: EMF, t_span: tuple, nt: int, rad: bool):
    cdef:
        int i, j, swtch
        double scaling, K, dt, temp = 0

        double p_plus[3]
        double p_minus[3]
        double p_prime[3]

        double tau[3]
        double sigma[3]

        double xtmp[3]
        double vtmp[3]

        double currE[3]
        double currH[3]

        double[:] t_mv
        double[:, :] x_mv
        double[:, :] p_mv
        double[:, :] v_mv
        double[:] tau_mv
        double[:] sigma_mv
        double[:] xtmp_mv
        double[:] p_minus_mv
        double[:] p_prime_mv
        double[:] vtmp_mv
        double[:] currE_mv
        double[:] currH_mv


# initialization
    t = np.linspace(t_span[0], t_span[1], nt)
    x = np.zeros((nt, 3), dtype=np.double)
    p = np.zeros((nt, 3), dtype=np.double)
    v = np.zeros((nt, 3), dtype=np.double)
    swtch = rad

    t_mv = t
    x_mv = x
    p_mv = p
    v_mv = v

    dt = t[1] - t[0]
    scaling = field.omg * ptcl.q ** 2 / ptcl.m
    K = 0

    x[0, :] = ptcl.x0
    p[0, :] = ptcl.p0

    tau_mv = tau
    sigma_mv = sigma
    xtmp_mv = xtmp
    vtmp_mv = vtmp
    currE_mv = currE
    currH_mv = currH
    p_minus_mv = p_minus
    p_prime_mv = p_prime

# main cycle
    for i in range(1, nt, 1):
        for j in range(3):
            xtmp_mv[j] = x_mv[i-1, j]
        for j in range(3):
            currE_mv[j] = field.e(xtmp_mv, t_mv[i], j)
            currH_mv[j] = field.h(xtmp_mv, t_mv[i], j)

        temp = dot(p_mv[i-1, :], p_mv[i-1, :])
        for j in range(3):
            tau[j] = currH[j] * dt / 2 / sqrt(1 + temp)
        for j in range(3):
            sigma[j] = 2 * tau[j] / (1 + dot(tau_mv, tau_mv))
        for j in range(3):
            p_minus[j] = p_mv[i-1, j] + currE[j] * dt / 2
        for j in range(3):
            p_prime[j] = p_minus[j] + cross(p_minus_mv, tau_mv, j)
        for j in range(3):
            p_plus[j] = p_minus[j] + cross(p_prime_mv, sigma_mv, j)
        for j in range(3):
            p_mv[i, j] = p_plus[j] + currE[j] * dt / 2

        for j in range(3):
            vtmp[j] = (p_mv[i, j] + p_mv[i-1, j]) / 2 / sqrt(1 + temp)
        for j in range(3):
            sigma[j] = currE[j] + cross(vtmp_mv, currH_mv, j)  # Lorentz force
            K = (1 + temp) * swtch * 0.0118 * scaling * (dot(sigma_mv, sigma_mv) - (dot(vtmp_mv, sigma_mv)) ** 2)  # letter K
        for j in range(3):
            p_mv[i, j] = p_mv[i, j] - dt * K * vtmp[j]
        for j in range(3):
            x_mv[i, j] = x_mv[i-1, j] + p_mv[i, j] * dt / sqrt(1 + temp)

    return t, x, p

#TODO: Limit interaction time by introducing external field envelope
