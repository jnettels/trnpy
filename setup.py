# -*- coding: utf-8 -*-
'''
TRNpy: Parallelized TRNSYS simulation with Python
=================================================

**Setup script for the TRNpy project**

Run with the following command prompt to install into Python environment:

.. code::

    python setup.py install


'''
from setuptools import setup
from setuptools_scm import get_version


try:
    version = get_version(version_scheme='post-release')
except LookupError:
    version = '0.0.0'
    print('Warning: setuptools-scm requires an intact git repository to detect'
          ' the version number for this build.')

print('Building TRNpy with version tag: ' + version)

# The setup function
setup(
    name='trnpy',
    version=version,
    description='Parallelized TRNSYS simulation with Python',
    long_description=open('README.md').read(),
    license='GPL-3.0',
    author='Joris Nettelstroth',
    author_email='joris.nettelstroth@stw.de',
    url='https://github.com/jnettels/trnpy',
    install_requires=['pandas>=0.24.1', ],
    python_requires='>=3.7',
    packages=['trnpy', 'trnpy/examples'],
    package_data={'trnpy/examples': ['Parametrics.xlsx'], },
    entry_points={
        'console_scripts': ['trnpy = trnpy.trnpy_script:main'],
        }
)
