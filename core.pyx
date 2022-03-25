# cython: language_level=3str
import numpy as np # to use numpy methods
import math as mt
from classes import EMF

cimport numpy as np # to convert numpy into c


cdef gamma(p: np.ndarray[3]):
    return mt.sqrt(1. + p.dot(p))

cdef lorentz(double q, np.ndarray v, np.ndarray e, np.ndarray h):
    return q*(e + np.cross(v, h))

#TODO: Introduce reaction force and modify boris() accordingly
'''
cdef radFrict(void):
    return -1
''' # radiation reaction force is not implemented

cpdef boris(p0: np.ndarray, x0: np.ndarray, charge: float, mass: float,
            field: EMF, t_span: tuple, nt: int):
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
        p_n_plus_half = p_minus + np.cross(p_minus + np.cross(p_minus, tau),
                                            2 * np.divide(tau, (1 + np.dot(tau, tau)))) \
                                + np.multiply(e, charge) * (dt / 2)

        p[j, :] = 0.5 * (p_n_plus_half + p_n_minus_half)
        if j != nt-1:
            r[j+1, :] = r[j, :] + dt * np.divide(p_n_plus_half, (mass * gamma(p_n_plus_half)))
        v[j, :] = np.divide(p[j, :], (mass * gamma(p[j, :])))

    return r, p, v, time

cpdef get_spectre(theta: float, phi: float, q: float, m: float, r: np.ndarray,
                  v: np.ndarray, time: np.ndarray, omg: np.ndarray):
    cdef:
        int N = time.size
        int omgN = omg.size
        np.ndarray n = np.array([mt.sin(theta)*mt.cos(phi), mt.sin(theta)*mt.sin(phi), mt.cos(theta)])
        double dt = time[1] - time[0]

        np.ndarray Xi = np.zeros(N)
        np.ndarray delXi = np.zeros(N-1)
        np.ndarray avgXi = np.zeros(N-1)

        np.ndarray delV = np.zeros((N-1, 3))
        np.ndarray avgV = np.zeros((N-1, 3))

        np.ndarray J
        np.ndarray temp = np.zeros(3)
        np.ndarray ansJ = np.zeros(omgN)

    for i in range(0, N-1, 1):
        Xi[i] = time[i] - np.dot(n, r[i, :])

    for i in range(0, N-2, 1):
        delXi[i] = Xi[i+1] - Xi[i]
        avgXi[i] = 0.5 * (Xi[i] + Xi[i+1])
        delV[i, :] = v[i+1, :] - v[i, :]
        avgV[i, :] = 0.5 * (v[i+1, :] + v[i, :])

    # Imaginary unit for exp()
    j = 0+1j

    # Main cycle
    for k in range(0, omgN-1, 1):
        J = np.zeros(3).astype(np.csingle)
        # Single integral calculation
        for i in range(0, N-2, 1):
            J += np.exp(j*omg[k]*avgXi[i]) * (dt/delXi[i]) * (
                2*mt.sin(0.5*omg[k]*delXi[i]) * avgV[i, :]
                + j*delV[i, :] * (
                    mt.sin(0.5*omg[k]*delXi[i]) / (0.5*omg[k]*delXi[i])
                    - mt.cos(0.5*omg[k]*delXi[i]))
            )

        # dE/dOmega
        temp = np.cross(n, np.cross(n, J))
        ansJ[k] = q**2 * omg[k] / (4*np.pi**2) * np.dot(temp, temp).real

    return ansJ

cpdef get_heatmap_data(phi: float, q: float, m: float, r: np.ndarray, v: np.ndarray,
                    time: np.ndarray, omg_span: tuple, nomg: int, ntheta: int):
    cdef:
        np.ndarray omg = np.linspace(omg_span[0], omg_span[1], nomg)
        np.ndarray theta = np.linspace(0, 2*np.pi, ntheta)
        int i = 0

        np.ndarray res = np.zeros((nomg, ntheta))
    for i in range(0, ntheta-1, 1):
        res[i, :] = get_spectre(theta[i], phi, q, m, r, v, time, omg)
    return res

#TODO: Limit interaction time by introducing external field envelope

#TODO: Add function to draw spectrum colormap for a given angle interval (all angles)
