import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl


class EMF:
    __slots__ = ['e', 'h']

    def __init__(self):
        self.e = None
        self.h = None

    def init_const(self, ampl=1.):
        def e(x, t):
            return ampl * np.array([0., 0., 0.]).astype(np.double)

        def h(x, t):
            return ampl * np.array([0., 0., 1.]).astype(np.double)

        self.e = e
        self.h = h

    def init_wave(self, ampl=1., omg=1., k=(0., 0., 1.), alph=0.):
        _k = np.array(k).astype(np.double)
        _k = _k / np.sqrt(_k.dot(_k))

        def e(x, t):
            ex = ampl * math.cos(omg * t - np.dot(_k, x) + alph)
            ey = ampl * math.sin(omg * t - np.dot(_k, x) + alph)
            ez = 0
            return np.array([ex, ey, ez]).astype(np.double)

        def h(x, t):
            return np.cross(e(x, t), _k)

        self.e = e
        self.h = h

    def init_gauss():
        print('err: Gaussian beam not implemented')
        return -1


class Trajectory:
    def __init__(self, r: np.ndarray, p: np.ndarray, v: np.ndarray, t: np.ndarray):
        self.r = r
        self.p = p
        self.v = v
        self.t = t

    def space(self):
        mpl.use('Qt5Agg')
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        z = self.r[:, 2]
        y = self.r[:, 1]
        x = self.r[:, 0]
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.plot3D(x, y, z)
        plt.show()

    def planes(self):
        z = self.r[:, 2]
        y = self.r[:, 1]
        x = self.r[:, 0]
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
        plt.show()

    def time(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        leg = ['t', ['x', 'y', 'z']]
        for i in range(3):
            plt.plot(self.t[:], self.r[:, i], label=leg[1][i])
        ax.set_xlabel(leg[0])
        plt.grid()
        plt.legend()
        plt.show()
