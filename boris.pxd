# cython: language_level=3str


cdef double dot(double[:] vec1, double[:] vec2) nogil

cdef double cross(double[:] vec1, double[:] vec2, int j) nogil

cdef class Particle:
    cdef:
        double q
        double m
        double[:] x0
        double[:] p0

cdef class EMF:
    cdef:
        int par

        double omg
        double alph
        double k[3]

        double[:] k_mv
    cdef double e(self, double[:] x, double t, int j) nogil
    cdef double h(self, double[:] x, double t, int j) nogil