'''
Setup script for the TRNpy project.

Run with the following command prompt to create a Windows executable:
python setup.py build
'''
from setuptools_scm import get_version
from cx_Freeze import setup, Executable
import os

version = get_version(version_scheme='post-release')

if 'g' in version:  # 'Dirty' version, does not fit to Windows' version scheme
    version_list = []
    for i in version.split('.'):
        try:  # Sort out all parts of the version name that are not integers
            version_list.append(str(int(i)))
        except Exception:
            pass
    if len(version_list) < 4:  # Windows version has a maximum length
        version_list.append('1')  # Use this to mark as a dev build
    version = '.'.join(version_list)

print('Building TRNpy with version tag: ' + version)

# These settings solved an error, but the paths are different for every user:
os.environ['TCL_LIBRARY'] = r'C:\Users\nettelstroth\Anaconda3\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\nettelstroth\Anaconda3\tcl\tk8.6'

# The setup function
setup(
    name='TRNpy',
    options={'build_exe': {'packages': ['idna', 'numpy']}},
    version=version,
    description='Parallelized TRNSYS simulation with Python',
    executables=[Executable('trnpy.py', base=None, icon='res/icon.ico')],
)