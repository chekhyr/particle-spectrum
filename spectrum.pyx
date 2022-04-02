# cython: language_level=3str, boundscheck=False, wraparound=False, cdivision=True
import numpy as np # to use numpy methods

cimport numpy as np # to convert numpy into c
from libc.math cimport sqrt, sin, cos
from boris cimport dot, Particle


cpdef spectrum_routine(t: np.ndarray, x: np.ndarray, p: np.ndarray, v: np.ndarray, direction: np.ndarray, omg_span: tuple, nOmg: int, ptcl: Particle):
    cdef:
        int i, j, k, stp
        double n[3]

        double dt
        double Xi, ntXi, dXi, averXi

        double dV[3]
        double averV[3]

        double realInteg[3]
        double imagInteg[3]
        double nrealInteg
        double nimagInteg

        double[:] n_mv
        double[:] t_mv
        double[:, :] x_mv
        double[:, :] p_mv
        double[:, :] v_mv

        double[:] omg_mv
        double[:] res_mv


    n = direction
    n_mv = n
    dt = sqrt(dot(n_mv, n_mv)) # abs(n), dt is reused for optimization
    for i in range(3):
        n[i] /= dt

    dt = t[1] - t[0]
    omg = np.linspace(omg_span[0], omg_span[1], nOmg)
    res = np.zeros(nOmg)

    t_mv = t
    x_mv = x
    p_mv = p
    v_mv = v

    omg_mv = omg
    res_mv = res
    stp = t.size - 2
    for j in range(0, nOmg-1, 1):
        for k in range(3):
            realInteg[k] = 0
            imagInteg[k] = 0
        for i in range(0, stp, 1):
            Xi = t_mv[i] - dot(n_mv, x_mv[i, :])
            ntXi = t_mv[i+1] - dot(n_mv, x_mv[i+1, :])
            dXi = ntXi - Xi
            averXi = 0.5 * (ntXi + Xi)
            for k in range(3):
                dV[k] = v_mv[i+1, k] - v_mv[i, k]
                averV[k] = 0.5 * (v_mv[i+1, k] + v_mv[i, k])

            for k in range(3):
                realInteg[k] += (dt/dXi) * ( 2*averV[k]*sin( omg_mv[j]*dXi*0.5 )*cos ( omg_mv[j]*averXi ) \
                                              - 2*dV[k]*sin( omg_mv[j]*dXi*0.5 )/omg_mv[j]/dXi*sin( omg_mv[j]*averXi ) \
                                              - dV[k]*cos( omg_mv[j]*dXi*0.5 )*sin( omg_mv[j]*averXi ) )
                imagInteg[k] += (dt/dXi) * ( 2*averV[k]*sin( omg_mv[j]*dXi*0.5 )*sin( omg_mv[j]*averXi ) \
                                                 - 2*dV[k]*sin( omg_mv[j]*dXi*0.5 )/omg_mv[j]/dXi*cos ( omg_mv[j]*averXi ) \
                                                 - dV[k]*cos(omg_mv[j] * dXi * 0.5)*cos ( omg_mv[j]*averXi ) )
        nrealInteg = dot(n_mv, realInteg)
        nimagInteg = dot(n_mv, imagInteg)
        res_mv[j] = dot(realInteg, realInteg) + dot(imagInteg, imagInteg) - nrealInteg**2 - nimagInteg**2
        res_mv[j] *= (0.5 * ptcl.q * omg[j] / np.pi) ** 2
    return omg, res