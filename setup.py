from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
            Extension( "src.*", 
                       [ "src/*.pyx" ], 
                       include_dirs=[numpy.get_include()] )
    ]

extensions = cythonize(extensions)

setup(
    name='rambutan',
    version='0.4.0',
    author='Jacob Schreiber',
    author_email='jmschr@cs.washington.edu',
    packages=['rambutan'],
    license='LICENSE.txt',
    description='Rambutan is a package for the analysis of the 3D structure of the genome.',
    ext_modules=extensions,
    install_requires=[
        "cython >= 0.22.1",
        "numpy >= 1.8.0",
        "joblib >= 0.9.0b4",
        "mxnet >= 0.5a3"
    ],
)
