# cython: language_level=3str, boundscheck=False, wraparound=False, cdivision=True
from __future__ import print_function
import numpy as np # to use numpy methods

cimport numpy as np # to convert numpy into c
from libc.math cimport sqrt, sin, cos
from boris cimport dot, cross


cpdef intensity_integral(t: np.ndarray, x: np.ndarray, p: np.ndarray, v: np.ndarray, omg_span: tuple, nOmg: int):
    cdef:
        int i, j, k, stp
        double dt
        double n[3]

        double vec1[3]
        double vec2[3]
        double vec3[3]
        double vec4[3]
        double vec5[3]
        double vec6[3]

        double ctXi, ntXi, dXi, averXi
        double ctTmp[3]
        double ntTmp[3]

        double dV[3]
        double averV[3]

        double[:] n_mv
        double[:] t_mv
        double[:, :] x_mv
        double[:, :] p_mv
        double[:, :] v_mv

        double[:] vec1_mv
        double[:] vec2_mv
        double[:] vec3_mv
        double[:] vec4_mv
        double[:] vec5_mv
        double[:] vec6_mv

        double[:] ctTmp_mv
        double[:] ntTmp_mv

        double[:] dV_mv
        double[:] averV_mv

        double[:] omg_mv
        double[:] res_mv

    dt = t[1] - t[0]
    n = (1, 0, 0)
    omg = np.linspace(omg_span[0], omg_span[1], nOmg)
    res = np.zeros(nOmg)

    n_mv = n
    t_mv = t
    x_mv = x
    p_mv = p
    v_mv = v

    vec1_mv = vec1
    vec2_mv = vec2
    vec3_mv = vec3
    vec4_mv = vec4
    vec5_mv = vec5
    vec6_mv = vec6

    ctTmp_mv = ctTmp
    ntTmp_mv = ntTmp

    dV_mv = dV
    averV_mv = averV

    omg_mv = omg
    res_mv = res
    stp = t.size - 1

    for i in range(1, nOmg, 1):
        for k in range(3):
            vec1_mv[k] = 0
            vec2_mv[k] = 0
        for j in range(1, stp, 1):
            ctXi = t_mv[j] - dot(n_mv, x_mv[j, :])
            ntXi = t_mv[j+1] - dot(n_mv, x_mv[j+1, :])
            dXi = ntXi - ctXi
            averXi = 0.5*(ntXi + ctXi)

            for k in range(3):
                dV[k] = v_mv[j+1, k] - v_mv[j, k]
                averV[k] = 0.5*(v_mv[j+1, k] + v_mv[j, k])
            for k in range(3):
                ctTmp[k] = cos(omg_mv[i] * averXi) * dt / dXi * 2 * averV_mv[k] * sin(omg_mv[i] * dXi / 2)  # 1 часть
                ctTmp[k] = ctTmp[k] - sin(omg_mv[i] * averXi) * dt / dXi * dV[k] * (
                            (sin(omg_mv[i] * dXi / 2) / (omg_mv[i] * dXi / 2)) - cos(omg_mv[i] * dXi / 2))

                ntTmp[k] = cos(omg_mv[i] * averXi) * dt / dXi * dV[k] * (
                            (sin(omg_mv[i] * dXi / 2) / (omg_mv[i] * dXi / 2)) - cos(omg_mv[i] * dXi / 2))
                ntTmp[k] = ntTmp[k] + sin(omg_mv[i] * averXi) * dt / dXi * 2 * averV_mv[k] * sin(omg_mv[i] * dXi / 2)
            for k in range(3):
                vec1_mv[k] = vec1_mv[k] + ctTmp_mv[k]
                vec2_mv[k] = vec2_mv[k] + ntTmp_mv[k]

        for k in range(3):
            vec3_mv[k] = cross(n_mv, vec1_mv, k)
            vec4_mv[k] = cross(n_mv, vec2_mv, k)
        for k in range(3):
            vec5_mv[k] = cross(n_mv, vec3_mv, k)
            vec6_mv[k] = cross(n_mv, vec4_mv, k)

        res_mv[i] = (dot(vec5_mv, vec5_mv) + dot(vec6_mv, vec6_mv)) * omg_mv[i]
    return omg, res