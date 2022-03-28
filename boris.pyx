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
        double dt, temp = 0

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
            sigma[j] = ptcl.q * (currE[j] + cross(vtmp, currH, j))  # Lorentz force
            K = (1 + temp) * 1.18 * 0.01 * (
                    dot(sigma, sigma) - (dot(vtmp, sigma)) ** 2)  # letter К
        if (rad == 0):
            K = 0 # radiation == 0
        for j in range(3):
            p[i, j] = p[i, j] - dt * K * vtmp[j]
        for j in range(3):
            x[i, j] = x[i-1, j] + p[i, j] * dt / sqrt(1 + temp)

    return (t, x, p)

'''
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
'''

#TODO: Limit interaction time by introducing external field envelope

#TODO: Add function to draw spectrum colormap for a given angle interval (all angles)
