import numpy as np # to use numpy methods

cimport numpy as np # to convert numpy into c
from libc.math cimport sqrt


cpdef intencity_integral(t: np.ndarray, x: np.ndarray, p: np.ndarray)
        cdef double dt
        cdef double gam, gam1
        cdef int i, j, k
        dt = self.time_mv[1] - self.time_mv[0]
        self.n_mv[0] = 1
        for i in range(1, self.size, 1):
            for k in range(3):
                self.vect1_mv[k] = 0
                self.vect2_mv[k] = 0
            for j in range(self.size_time - 1):
                for k in range(3):
                    self.r_mv[k] = self.trajec_mv[j][k]
                    self.r1_mv[k] = self.trajec_mv[j+1][k]
                    self.v_mv[k] = self.momen_mv[j][k]
                    self.v1_mv[k] = self.momen_mv[j+1][k]
                self.ksi = self.time_mv[j] - scalar_prod(self.n_mv, self.r_mv)
                self.ksi1 = self.time_mv[j+1] - scalar_prod(self.n_mv, self.r1_mv)
                self.dksi = self.ksi1 - self.ksi
                self.ksi_ave = (self.ksi1 + self.ksi) / 2

                gam = 1 + scalar_prod(self.v_mv, self.v_mv)
                gam1 = 1 + scalar_prod(self.v1_mv, self.v1_mv)
                for k in range(3):
                    self.v_mv[k] = self.v_mv[k] / sqrt(gam)
                    self.v1_mv[k] = self.v1_mv[k] / sqrt(gam1)
                for k in range(3):
                    self.dv_mv[k] = self.v1_mv[k] - self.v_mv[k]
                    self.v_ave_mv[k] = (self.v1_mv[k] + self.v_mv[k]) / 2
                for k in range(3):
                    self.temp_mv[k] = cos(self.omega_mv[i] * self.ksi_ave) * dt / self.dksi * 2 * self.v_ave_mv[k] * sin(self.omega_mv[i] * self.dksi / 2)  # 1 часть
                    self.temp_mv[k] = self.temp_mv[k] - sin(self.omega_mv[i] * self.ksi_ave) * dt / self.dksi * self.dv_mv[k] * ((sin(self.omega_mv[i] * self.dksi / 2) / (self.omega_mv[i] * self.dksi / 2))  - cos(self.omega_mv[i] * self.dksi / 2))

                    self.temp1_mv[k] = cos(self.omega_mv[i] * self.ksi_ave) * dt / self.dksi * self.dv_mv[k] * ((sin(self.omega_mv[i] * self.dksi / 2) / (self.omega_mv[i] * self.dksi / 2))  - cos(self.omega_mv[i] * self.dksi / 2))
                    self.temp1_mv[k] = self.temp1_mv[k] + sin(self.omega_mv[i] * self.ksi_ave) * dt / self.dksi * 2 * self.v_ave_mv[k] * sin(self.omega_mv[i] * self.dksi / 2)

                for k in range(3):
                    self.vect1_mv[k] = self.vect1_mv[k] + self.temp_mv[k]
                    self.vect2_mv[k] = self.vect2_mv[k] + self.temp1_mv[k]

            for k in range(3):
                self.vect3_mv[k] = vector_prod(self.n_mv, self.vect1_mv, k)
                self.vect4_mv[k] = vector_prod(self.n_mv, self.vect2_mv, k)
            for k in range(3):
                self.vect5_mv[k] = vector_prod(self.n_mv, self.vect3_mv, k)
                self.vect6_mv[k] = vector_prod(self.n_mv, self.vect4_mv, k)

            self.result_mv[i] = (scalar_prod(self.vect5_mv, self.vect5_mv) + scalar_prod(self.vect6_mv, self.vect6_mv)) * self.omega_mv[i]
        return self.result


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
