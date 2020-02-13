from distutils.core import setup
#from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy

setup(
    ext_modules=cythonize(['main.pyx'],
    compiler_directives={'boundscheck': False, 'wraparound': False}),
    include_dirs=[numpy.get_include()]
)
