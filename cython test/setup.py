from setuptools import setup, Extension
from Cython.Build import cythonize

#module = Extension ('lear-test', sources=['learntest1.pyx'])

setup(ext_modules=cythonize('primer.pyx'))
