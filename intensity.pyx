# cython: language_level=3str, boundscheck=False, wraparound=False, cdivision=True
from __future__ import print_function
import numpy as np # to use numpy methods

cimport numpy as np # to convert numpy into c
from libc.math cimport sqrt, sin, cos
from boris cimport dot, cross


cpdef intensity_integral(t: np.ndarray, x: np.ndarray, p: np.ndarray, v: np.ndarray, direction: np.ndarray, omg_span: tuple, nOmg: int):
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
        double tmp[3]

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
        double[:] tmp_mv

        double[:] dV_mv
        double[:] averV_mv

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
#TODO: Implement proper algorithm (see paper notes)
    for j in range(1, nOmg, 1):
        for i in range(1, stp, 1):

    return omg, res