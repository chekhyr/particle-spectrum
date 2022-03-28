# cython: language_level=3str, boundscheck=False, wraparound=False, cdivision=True
import numpy as np # to use numpy methods
import matplotlib as mpl
import matplotlib.pyplot as plt

cimport numpy as np # to convert numpy into c
from libc.math cimport sin, cos, sqrt

cdef double pi = np.pi
cdef enum:
    nt = 10000

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
    if j == 0:
        return vec1[1] * vec2[2] - vec1[2] * vec2[1]
    elif j == 1:
        return vec1[2] * vec2[0] - vec1[0] * vec2[2]
    elif j == 2:
        return vec1[0] * vec2[1] - vec1[1] * vec2[0]

cdef class EMF:
    cdef:
        char* par
        double k[3]
        double tmp[3]
        double omg
        double alph

    def __init__(self, char* par):
        self.par = par
        if self.par != b'const':
            k = (0, 0, 1)
            omg = 1
            alph = 0

    cdef double e(self, x, t, j):
        if self.par == b'const':
            return 0

        elif self.par == b'wave':
            if j == 0:
                return cos(self.omg * t - dot(self.k, x) + self.alph)
            elif j==1:
                return sin(self.omg * t - dot(self.k, x) + self.alph)
            elif j==2:
                return 0

        elif self.par == b'gauss':
            raise NotImplementedError


    cdef double h(self, x, t, j):

        if self.par == b'const':
            if j==2:
                return 1
            else:
                return 0

        elif self.par == b'wave':
            for i in range(3):
                self.tmp[i] = self.e(x, t, i)
            return cross(self.tmp, self.k, j)

        elif self.par == b'gauss':
            raise NotImplementedError

cdef double gamma(double[:] p):
    return sqrt(1. + dot(p, p))

'''
cdef lorentz(double q, double[:] v, double[:] e, double[:] h, int j):
    return q*(e[j] + cross(v, h, j))
''' # deprecated

#TODO: Introduce reaction force and modify boris() accordingly
'''
cdef radFrict(void):
    return -1
''' # radiation reaction force is not implemented

ctypedef struct Trajectory:
    double t[nt]
    double x[nt][3]
    double p[nt][3]

cdef Trajectory boris(Particle ptcl, EMF field, (double, double) t_span):
    cdef:
    # plot arrays
        Trajectory res

    # locals
        int i, j
        double dt, temp = 0

        double[:, :] p_mv = res.p


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
    res.t = np.linspace(t_span[0], t_span[1], nt)
    dt = res.t[1] - res.t[0]

    res.x[0][:] = ptcl.x0
    res.p[0][:] = ptcl.p0

# main cycle
    for i in range(1, nt, 1):
        for j in range(3):
            xtmp[j] = res.x[i-1][j]
            currE[j] = field.e(xtmp, res.t[i], j)
            currH[j] = field.h(xtmp, res.t[i], j)

        temp = dot(p_mv[i-1, :], p_mv[i-1, :])
        for j in range(3):
            tau[j] = currH[j] * dt / 2 / sqrt(1 + temp)
        for j in range(3):
            sigma[j] = 2 * tau[j] / (1 + dot(tau, tau))
        for j in range(3):
            p_minus[j] = res.p[i-1][j] + currE[j] * dt / 2
        for j in range(3):
            p_prime[j] = p_minus[j] + cross(p_minus, tau, j)
        for j in range(3):
            p_plus[j] = p_minus[j] + cross(p_prime, sigma, j)
        for j in range(3):
            res.p[i][j] = p_plus[j] + currE[j] * dt / 2

        for j in range(3):
            vtmp[j] = (res.p[i][j] + res.p[i - 1][j]) / 2 / sqrt(1 + temp)
        for j in range(3):
            sigma[j] = currE[j] + cross(vtmp, currH, j)  # Lorentz force
            K = (1 + temp) * 1.18 * 0.01 * (
                    dot(sigma, sigma) - (dot(vtmp, sigma)) ** 2)  # letter К
        #K = 0 # radiation == 0
        for j in range(3):
            res.p[i][j] = res.p[i][j] - dt * K * vtmp[j]
        for j in range(3):
            res.x[i][j] = res.x[i - 1][j] + res.p[i][j] * dt / sqrt(1 + temp)

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
ptcl.p0 = (0.1, 0, 1)

objEMF = EMF(b'const')

cdef Trajectory trj
trj = boris(ptcl, objEMF, (0, 600))
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
