from setuptools import setup, Extension

module = Extension ('example', sources=['learn-risk-1.pyx'])

setup(
    name='cythonTest',
    version='1.0',
    author='branexius',
    ext_modules=[module]
)