#!/usr/bin/env python
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        'intensity', ['intensity.pyx', 'boris.pyx'],
        include_dirs=[numpy.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    ),
]


setup(
    ext_modules=cythonize(extensions)
)
