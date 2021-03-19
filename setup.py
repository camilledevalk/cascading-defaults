from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

with open('README.md', 'r') as file:
    long_description = file.read()
    
ext_modules = [
    Extension('cython_defaults', ['cascading_defaults/simulation/defaults.pyx']),
    Extension('cython_sorting', ['cascading_defaults/utils/sorting.pyx'])
]

setup(
    name='cascading_defaults',
    description='Package for research on cascading defaults',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.6',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
    install_requires=[
        'cython',
        'cycler',
        'numpy',
        'matplotlib',
        'pandas',
        'scipy'
    ],
    version='0.0.1'
)