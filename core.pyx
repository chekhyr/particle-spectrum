# cython: language_level=3str, boundscheck=False, wraparound=False, cdivision=True
import numpy as np # to use numpy methods
import matplotlib as mpl
import matplotlib.pyplot as plt

cimport numpy as np # to convert numpy into c
from libc.math cimport sin, cos, sqrt

cdef double pi = np.pi
cdef enum:
    nt = 100

ctypedef struct Particle:
    double q
    double m
    double x0[3]
    double p0[3]

cdef double dot(double[:] vec1, double[:] vec2):
    cdef double res = 0
    for i in range(3):
        res += vec1[i]*vec2[i]
    return res

cdef double cross(double[:] vec1, double[:] vec2, int j):
    if j == 1:
        return vec1[2] * vec2[3] - vec1[3] * vec2[2]
    elif j == 2:
        return vec1[3] * vec2[1] - vec1[1] * vec2[3]
    elif j == 3:
        return vec1[1] * vec2[2] - vec1[2] * vec2[1]

cdef class EMF:
    cdef:
        char* par
        double[:] k_mv

    def __init__(self, char* par):
        self.par = par
        k_mv = np.ndarray((0, 0, 1), dtype=np.double)
        #self.k_mv = np.divide(k, np.sqrt(k.dot(k)))

    def e(self, x, t):
        res = [0, 0, 0]

        if self.par == b'const':
            for i in range(3):
                res[i] = 0

        elif self.par == b'wave':
            omg = 1.
            alph = 0.

            res[0] = cos(omg * t - dot(self.k_mv, x) + alph)
            res[1] = sin(omg * t - dot(self.k_mv, x) + alph)
            res[2] = 0

        elif self.par == b'gauss':
            raise NotImplementedError

        return np.array((res[0], res[1], res[2]), dtype=np.double)

    def h(self, x, t):
        res = [0, 0, 0]

        if self.par == b'const':
            for i in range(2):
                res[i] = 0
            else:
                res[2] = 1

        elif self.par == b'wave':
            res[:] = np.cross(self.e(x, t), self.k_mv)

        elif self.par == b'gauss':
            raise NotImplementedError

        return np.array((res[0], res[1], res[2]), dtype=np.double)

cdef double gamma(double[:] p):
    return sqrt(1. + dot(p, p))

cdef lorentz(double q, double[:] v, double[:] e, double[:] h, int j):
    return q*(e[j] + cross(v, h, j))

#TODO: Introduce reaction force and modify boris() accordingly
'''
cdef radFrict(void):
    return -1
''' # radiation reaction force is not implemented
ctypedef struct Trajectory:
    double t[nt]
    double x[nt][3]
    double p[nt][3]
    double v[nt][3]

cdef Trajectory boris(Particle ptcl, EMF field, (double, double) t_span):
    cdef:
    # plot arrays
        Trajectory res

    # locals
        int i, j
        double[:] e_mv
        double[:] h_mv
        double dt

        double[:] t_mv = res.t
        double[:, :] x_mv = res.x
        double[:, :] p_mv = res.p
        double[:, :] v_mv = res.v

        double p_n_plus_half[3]
        double p_n_minus_half[3]
        double p_minus[3]
        double tau[3]

        double[:] p_n_plus_half_mv
        double[:] p_n_minus_half_mv
        double[:] p_minus_mv
        double[:] tau_mv

    p_n_plus_half_mv = p_n_plus_half
    p_n_minus_half_mv = p_n_minus_half
    p_minus_mv = p_minus
    tau_mv = tau

# initialization
    res.t = np.linspace(t_span[0], t_span[1], nt)
    dt = res.t[1] - res.t[0]

    res.x[0][:] = ptcl.x0
    res.p[0][:] = ptcl.p0
    for i in range(3):
        res.v[0][i] = ptcl.p0[i]/(ptcl.m*gamma(ptcl.p0))

    e_mv = field.e(ptcl.x0, res.t[0])
    h_mv = field.h(ptcl.x0, res.t[0])

# main cycle
    for i in range(3):
        p_n_plus_half[i] = ptcl.p0[i] + (dt / 2) * lorentz(ptcl.q, v_mv[0, :], e_mv, h_mv, i)
    for i in range(3):
        x_mv[1, i] = x_mv[0, i] + dt * p_n_plus_half[i] / (ptcl.m * gamma(p_n_plus_half))

    cdef:
        double tmp1[3]
        double tmp2[3]

        double[:] tmp1_mv
        double[:] tmp2_mv

    tmp1_mv = tmp1
    tmp2_mv = tmp2
    for j in range(1, nt-1, 1):
        e_mv = field.e(x_mv[j, :], res.t[j])
        h_mv = field.h(x_mv[j, :], res.t[j])

        p_n_minus_half = p_n_plus_half
        for i in range(3):
            p_minus[i] = p_n_minus_half[i] + e_mv[i] * ptcl.q * (dt / 2)
        for i in range(3):
            tau[i] = h_mv[i] * ptcl.q / (ptcl.m * gamma(p_minus_mv[:]) * (dt / 2))
        for i in range(3):
            tmp1[i] = p_minus[i] + cross(p_minus_mv[:], tau_mv[:], i)
            tmp2[i] = 2 * tau[i] / (1 + dot(tau_mv[:], tau_mv[:]))
        for i in range(3):
            p_n_plus_half[i] = p_minus[i] + cross(tmp1_mv[:], tmp2_mv[:], i) + e_mv[i] * ptcl.q * (dt / 2)

        for i in range(3):
            p_mv[j, i] = 0.5 * (p_n_plus_half[i] + p_n_minus_half[i])
        if j != nt-1:
            for i in range(3):
                x_mv[j+1, i] = x_mv[j, i] + dt * p_n_plus_half[i] / (ptcl.m * gamma(p_n_plus_half_mv[:]))

        for i in range(3):
            v_mv[j, i] = p_mv[j, i] / (ptcl.m * gamma(p_mv[j, :]))

    return res

class Plots:
    __slots__ = ['x', 't']

    def __init__(self, double[:, :] x, double[:] t):
        self.x = x
        self.t = t
        mpl.use('QtCairo')

    def space(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        z = self.x[:, 2]
        y = self.x[:, 1]
        x = self.x[:, 0]

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.plot3D(x, y, z)

        plt.tight_layout()
        plt.show()

    def planes(self):
        z = self.x[:, 2]
        y = self.x[:, 1]
        x = self.x[:, 0]

        plt.subplot(2, 2, 1)
        plt.plot(x, y)
        plt.title("xy", loc='left', y=0.85)
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.plot(x, z)
        plt.title("xz", loc='left', y=0.85)
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.plot(y, z)
        plt.title("yz", loc='left', y=0.85)
        plt.grid()

        plt.tight_layout()
        plt.show()

    def involute(self):
        plt.plot(self.t[:], self.x[:, 0], label='x')
        plt.plot(self.t[:], self.x[:, 1], label='y')
        plt.plot(self.t[:], self.x[:, 2], label='z')

        plt.xlabel('t')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()

cdef Particle ptcl
ptcl.q = 1
ptcl.m = 1
ptcl.x0 = (0, 0, 0)
ptcl.p0 = (0, 1, 1)

objEMF = EMF(b'const')

cdef Trajectory trj
trj = boris(ptcl, objEMF, (0, 1000))
cdef:
    double[:] t_mv
    double[:, :] x_mv
t_mv = trj.t
x_mv = trj.x
objPlt = Plots(x_mv, t_mv)
objPlt.space()

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
