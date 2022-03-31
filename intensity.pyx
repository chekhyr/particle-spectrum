# cython: language_level=3str, boundscheck=False, wraparound=False, cdivision=True
from __future__ import print_function
import numpy as np # to use numpy methods

cimport numpy as np # to convert numpy into c
from libc.math cimport sqrt, sin, cos
from boris cimport dot, cross


cpdef intensity_integral(t: np.ndarray, x: np.ndarray, p: np.ndarray, nparam: int):
    cdef:
        int i, j, k
        double dt
        double n[3]
        double vec1[3], vec2[3]
        double vec3[3], vec4[3], vec5[3], vec6[3]
        double ctX[3], ntX[3]
        double ctV[3], ntV[3], dV[3], averV[3]
        double ctXi, ntXi, dXi, averXi
        double ctGamma, ntGamma
        double ctTmp[3], ntTmp[3]

        double[:, :] x_mv, p_mv

    dt = t[1] - t[0]
    n = (1, 0, 0)
    omega = np.linspace(0, 5, nparam)
    res = np.zeros(nparam)

    x_mv = x
    p_mv = p

    for i in range(1, nparam, 1):
        for k in range(3):
            vec1[k] = 0
            vec2[k] = 0
        for j in range(t.size - 1):
            for k in range(3):
                ctX[k] = x_mv[j, k]
                ntX[k] = x_mv[j+1, k]
                ctV[k] = p_mv[j, k]
                ntV[k] = p_mv[j+1, k]
            ctXi = t[j] - dot(n, ctX)
            ntXi = t[j+1] - dot(n, ntX)
            dXi = ntXi - ctXi
            averXi = 0.5*(ntXi + ctXi)

            ctGamma = sqrt(1 + dot(n, ctX))
            ntGamma = sqrt(1 + dot(n, ntX))
            for k in range(3):
                ctV[k] /= ctGamma
                ntV[k] /= ntGamma
            for k in range(3):
                dV[k] = ntV[k] - ctV[k]
                averV[k] = 0.5*(ntV[k] + ctV[k])
            ctTmp = np.array((0, 0, 0))
            ntTmp = np.array((0, 0, 0))
            for k in range(3):
                ctTmp[k] = cos(omega[i] * averXi) * dt / dXi * 2 * averV[k] * sin(omega[i] * dXi / 2)  # 1 часть
                ctTmp[k] = ctTmp[k] - sin(omega[i] * averXi) * dt / dXi * dV[k] * (
                            (sin(omega[i] * dXi / 2) / (omega[i] * dXi / 2)) - cos(omega[i] * dXi / 2))

                ntTmp[k] = cos(omega[i] * averXi) * dt / dXi * dV[k] * (
                            (sin(omega[i] * dXi / 2) / (omega[i] * dXi / 2)) - cos(omega[i] * dXi / 2))
                ntTmp[k] = ntTmp[k] + sin(omega[i] * averXi) * dt / dXi * 2 * averV[k] * sin(omega[i] * dXi / 2)
            for k in range(3):
                vec1[k] = vec1[k] + ctTmp[k]
                vec2[k] = vec2[k] + ntTmp[k]

        for k in range(3):
            vec3[k] = cross(n, vec1, k)
            vec4[k] = cross(n, vec2, k)
        for k in range(3):
            vec5[k] = cross(n, vec3, k)
            vec6[k] = cross(n, vec4, k)

        res[i] = (dot(vec5, vec5) + dot(vec6, vec6)) * omega[i]
    return omega, res