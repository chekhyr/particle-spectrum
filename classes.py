import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



class Particle:
    __slots__ = ['q', 'm', 'x0', 'p0']

    def __init__(self, q: float, m: float, x0: np.ndarray, p0: np.ndarray):
        self.q = q
        self.m = m
        self.x0 = x0
        self.p0 = p0

'''
class EMF:
    __slots__ = ['e', 'h', 'ampl', 'omg']

    def __init__(self, ampl=1.):
        self.ampl = ampl
        self.e = None
        self.h = None
        self.omg = None

    def init_const(self):
        const = self.ampl

        def e(x, t):
            return const * np.array([0., 0., 0.]).astype(np.double)

        def h(x, t):
            return const * np.array([0., 0., 1.]).astype(np.double)

        self.e = e
        self.h = h
        self.omg = 0.

    def init_wave(self, k=(0., 0., 1.), alph=0.):
        _k = np.array(k).astype(np.double)
        omg = self.omg = _k.dot(_k)  # N.B. c = 1
        _k = _k / np.sqrt(omg)

        def e(x, t):
            ex = self.ampl * np.cos(omg * t - np.dot(_k, x) + alph)
            ey = self.ampl * np.sin(omg * t - np.dot(_k, x) + alph)
            ez = 0
            return np.array([ex, ey, ez]).astype(np.double)

        def h(x, t):
            return np.cross(e(x, t), _k)

        self.e = e
        self.h = h

    def init_gauss(self):
        raise NotImplementedError
''' # deprecated


class Plots:
    __slots__ = ['x', 't']

    def __init__(self, t: np.ndarray, x: np.ndarray):
        self.t = t
        self.x = x
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

        ax.plot3D(x, y, z, color='hotpink')

        plt.tight_layout()
        plt.show()

    def planes(self):
        z = self.x[:, 2]
        y = self.x[:, 1]
        x = self.x[:, 0]

        plt.subplot(2, 2, 1)
        plt.plot(x, y, color='hotpink')
        plt.title("xy", loc='left', y=0.85)
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.plot(x, z, color='hotpink')
        plt.title("xz", loc='left', y=0.85)
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.plot(y, z, color='hotpink')
        plt.title("yz", loc='left', y=0.85)
        plt.grid()

        plt.tight_layout()
        plt.show()

    def involute(self):
        plt.plot(self.t[:], self.x[:, 0], label='x', c='pink')
        plt.plot(self.t[:], self.x[:, 1], label='y', c='hotpink')
        plt.plot(self.t[:], self.x[:, 2], label='z', c='magenta')

        plt.xlabel('t')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()

class PlotIntensity:
    __slots__ = ['omg', 'J']

    def __init__(self, omg: np.ndarray, J: np.ndarray):
        self.omg = omg
        self.J = J
        mpl.use('QtCairo')

    def draw(self):
        plt.plot(self.omg, self.J, label='J', c='hotpink')

        plt.xlabel('\omega')
        plt.grid()
        #plt.legend()

        plt.tight_layout()
        plt.show()