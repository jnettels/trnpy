# -*- coding: utf-8 -*-
'''
TRNpy: Parallelized TRNSYS simulation with Python
=================================================

**Setup script for the TRNpy project**

Run with the following command prompt to create a Windows executable:

.. code::

    python setup.py build

Lots of modules are excluded from the build and some unnecessary folders are
removed from the resulting folder. This can cause the program to fail, but
also dramatically decreases the folder size.
'''

from setuptools_scm import get_version
from cx_Freeze import setup, Executable
import os
import shutil


try:
    version = get_version(version_scheme='post-release')
except LookupError:
    version = '0.0.0.0'
    print('Warning: setuptools-scm requires an intact git repository to detect'
          ' the version number for this build.')

if 'g' in version:  # 'Dirty' version, does not fit to Windows' version scheme
    version_list = []
    for i in version.split('.'):
        try:  # Sort out all parts of the version name that are not integers
            version_list.append(str(int(i)))
        except Exception:
            pass
    if len(version_list) < 3:  # Version is X.Y -> make it X.Y.0.1
        version_list.append('0.1')  # Use this to mark as a dev build
    elif len(version_list) < 4:  # Windows version has a maximum length
        version_list.append('1')  # Use this to mark as a dev build
    version = '.'.join(version_list)

print('Building TRNpy with version tag: ' + version)

# These settings solved an error, but the paths are different for every user:
os.environ['TCL_LIBRARY'] = r'C:\Users\nettelstroth\Anaconda3\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\nettelstroth\Anaconda3\tcl\tk8.6'

# The setup function
setup(
    name='TRNpy',
    options={'build_exe': {'packages': ['numpy', 'asyncio'],
                           'includes': [],
                           'excludes': ['adodbapi',
                                        'alabaster'
                                        'asn1crypto',
                                        'babel',
                                        'backports',
                                        'bokeh',
                                        'bottleneck',
                                        'bs4',
                                        'certifi',
                                        'cffi',
                                        'chardet',
                                        'cloudpickle',
                                        'colorama',
                                        'concurrent',
                                        'cryptography',
                                        'ctypes',
                                        'curses',
                                        'Cython',
                                        'cytoolz',
                                        'dask',
                                        'et_xmlfile',
                                        'h5netcdf',
                                        'h5py',
                                        'html',
                                        'html5lib',
                                        'ipykernel',
                                        'IPython',
                                        'ipython_genutils',
                                        'jedi',
                                        'jinja2',
                                        'jupyter_client',
                                        'jupyter_core',
                                        'lib2to3',
                                        'lxml',
                                        'markupsafe',
                                        'matplotlib',
                                        'msgpack',
                                        'nbconvert',
                                        'nbformat',
                                        'netCDF4',
                                        'nose',
                                        'notebook',
                                        'numexpr',
                                        'openpyxl',
                                        'OpenSSL',
                                        'PIL',
                                        'pkg_resources',
                                        'prompt_toolkit',
                                        'pycparser',
                                        'pydoc_data',
                                        'pygments',
                                        'PyQt5',
                                        'requests',
                                        'scipy',
                                        'seaborn',
                                        'setuptools',
                                        'sphinx',
                                        'sphinxcontrib',
                                        'sqlalchemy',
                                        'sqlite3',
                                        'tables',
                                        'testpath',
                                        'tornado',
                                        'traitlets',
                                        'wcwidth',
                                        'webencodings',
                                        'win32com',
                                        'zict',
                                        'zmq',
                                        '_pytest',
                                        ],
                           }},
    version=version,
    executables=[Executable('trnpy.py', base=None, icon='res/icon.ico')],
    description='Parallelized TRNSYS simulation with Python',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license='BSD-3-Clause',
    author='Joris Nettelstroth',
    author_email='joris.nettelstroth@stw.de',
    url='https://bitbucket.org/joris_nettelstroth/trnpy',
)

# Remove some more specific folders:
remove_folders = [
        r'.\build\exe.win-amd64-3.6\mpl-data',
        r'.\build\exe.win-amd64-3.6\tk\demos',
        r'.\build\exe.win-amd64-3.6\tcl\tzdata',
        r'.\build\exe.win-amd64-3.6\lib\pandas\tests',
        ]
for folder in remove_folders:
    shutil.rmtree(folder, ignore_errors=True)

# Copy the README.md file to the build folder, changing extension to .txt
shutil.copy2(r'.\README.md', r'.\build\exe.win-amd64-3.6\README.txt')
