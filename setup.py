# Copyright (C) 2020 Joris Zimmermann

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see https://www.gnu.org/licenses/.

"""TRNpy: Parallelized TRNSYS simulation with Python.

TRNpy: Parallelized TRNSYS simulation with Python
=================================================
Simulate TRNSYS deck files in serial or parallel and use parametric tables to
perform simulations for different sets of parameters. TRNpy helps to automate
these and similar operations by providing functions to manipulate deck files
and run TRNSYS simulations from a programmatic level.

Module Setup
------------

Run with the following command prompt to install into Python environment:

.. code::

    pip install .

Or use the following to install all optional dependencies:

.. code::

    pip install .[full]


"""
from setuptools import setup

# The setup function
setup(
    name='trnpy',
    use_scm_version=True,
    description='Parallelized TRNSYS simulation with Python',
    long_description=open('README.md').read(),
    license='GPL-3.0',
    author='Joris Zimmermann',
    author_email='joris.zimmermann@stw.de',
    url='https://github.com/jnettels/trnpy',
    install_requires=['pandas>=2.0', 'psutil', 'openpyxl'],
    python_requires='>=3.7',
    setup_requires=['setuptools_scm'],
    extras_require={
        'full':  ['pyyaml', 'matplotlib', 'bokeh'],
    },
    packages=['trnpy', 'trnpy/examples'],
    package_data={'trnpy/examples': ['Parametrics.xlsx'], },
    entry_points={
        'console_scripts': ['trnpy = trnpy.trnpy_script:main'],
        }
)
