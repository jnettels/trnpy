'''
Setup script for the TRNpy project.

Run with the following command prompt to create a Windows executable:
python setup.py build
'''
from setuptools_scm import get_version
from cx_Freeze import setup, Executable
import os
import shutil


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
    options={'build_exe': {'packages': ['numpy']}},
    version=version,
    description='Parallelized TRNSYS simulation with Python',
    executables=[Executable('trnpy.py', base=None, icon='res/icon.ico')],
)

'''
Remove unnecessary folders from the resulting build. This can cause serious
issues, but also dramatically decreases the folder size.
'''
keep_folders = [
#        r'.\build\exe.win-amd64-3.6\lib\adodbapi',
#        r'.\build\exe.win-amd64-3.6\lib\alabaster',
#        r'.\build\exe.win-amd64-3.6\lib\asn1crypto',
#        r'.\build\exe.win-amd64-3.6\lib\asyncio',
#        r'.\build\exe.win-amd64-3.6\lib\babel',
#        r'.\build\exe.win-amd64-3.6\lib\backports',
#        r'.\build\exe.win-amd64-3.6\lib\bottleneck',
#        r'.\build\exe.win-amd64-3.6\lib\bs4',
#        r'.\build\exe.win-amd64-3.6\lib\certifi',
#        r'.\build\exe.win-amd64-3.6\lib\cffi',
#        r'.\build\exe.win-amd64-3.6\lib\chardet',
#        r'.\build\exe.win-amd64-3.6\lib\cloudpickle',
        r'.\build\exe.win-amd64-3.6\lib\collections',
#        r'.\build\exe.win-amd64-3.6\lib\colorama',
#        r'.\build\exe.win-amd64-3.6\lib\concurrent',
#        r'.\build\exe.win-amd64-3.6\lib\cryptography',
#        r'.\build\exe.win-amd64-3.6\lib\ctypes',
#        r'.\build\exe.win-amd64-3.6\lib\curses',
#        r'.\build\exe.win-amd64-3.6\lib\Cython',
        r'.\build\exe.win-amd64-3.6\lib\dateutil',
        r'.\build\exe.win-amd64-3.6\lib\distutils',
        r'.\build\exe.win-amd64-3.6\lib\docutils',
        r'.\build\exe.win-amd64-3.6\lib\email',
        r'.\build\exe.win-amd64-3.6\lib\encodings',
#        r'.\build\exe.win-amd64-3.6\lib\et_xmlfile',
#        r'.\build\exe.win-amd64-3.6\lib\html',
#        r'.\build\exe.win-amd64-3.6\lib\html5lib',
        r'.\build\exe.win-amd64-3.6\lib\http',
        r'.\build\exe.win-amd64-3.6\lib\imagesize',
        r'.\build\exe.win-amd64-3.6\lib\importlib',
#        r'.\build\exe.win-amd64-3.6\lib\ipykernel',
#        r'.\build\exe.win-amd64-3.6\lib\IPython',
#        r'.\build\exe.win-amd64-3.6\lib\ipython_genutils',
#        r'.\build\exe.win-amd64-3.6\lib\jedi',
#        r'.\build\exe.win-amd64-3.6\lib\jinja2',
        r'.\build\exe.win-amd64-3.6\lib\json',
        r'.\build\exe.win-amd64-3.6\lib\jsonschema',
#        r'.\build\exe.win-amd64-3.6\lib\jupyter_client',
#        r'.\build\exe.win-amd64-3.6\lib\jupyter_core',
#        r'.\build\exe.win-amd64-3.6\lib\lib2to3',
        r'.\build\exe.win-amd64-3.6\lib\logging',
#        r'.\build\exe.win-amd64-3.6\lib\lxml',
#        r'.\build\exe.win-amd64-3.6\lib\markupsafe',
#        r'.\build\exe.win-amd64-3.6\lib\matplotlib',
        r'.\build\exe.win-amd64-3.6\lib\multiprocessing',
#        r'.\build\exe.win-amd64-3.6\lib\nbconvert',
#        r'.\build\exe.win-amd64-3.6\lib\nbformat',
#        r'.\build\exe.win-amd64-3.6\lib\nose',
#        r'.\build\exe.win-amd64-3.6\lib\notebook',
        r'.\build\exe.win-amd64-3.6\lib\numexpr',
        r'.\build\exe.win-amd64-3.6\lib\numpy',
        r'.\build\exe.win-amd64-3.6\lib\numpydoc',
#        r'.\build\exe.win-amd64-3.6\lib\openpyxl',
#        r'.\build\exe.win-amd64-3.6\lib\OpenSSL',
        r'.\build\exe.win-amd64-3.6\lib\pandas',
#        r'.\build\exe.win-amd64-3.6\lib\PIL',
#        r'.\build\exe.win-amd64-3.6\lib\pkg_resources',
#        r'.\build\exe.win-amd64-3.6\lib\prompt_toolkit',
        r'.\build\exe.win-amd64-3.6\lib\psutil',
        r'.\build\exe.win-amd64-3.6\lib\py',
#        r'.\build\exe.win-amd64-3.6\lib\pycparser',
#        r'.\build\exe.win-amd64-3.6\lib\pydoc_data',
#        r'.\build\exe.win-amd64-3.6\lib\pygments',
#        r'.\build\exe.win-amd64-3.6\lib\PyQt5',
        r'.\build\exe.win-amd64-3.6\lib\pytz',
        r'.\build\exe.win-amd64-3.6\lib\pywin',
        r'.\build\exe.win-amd64-3.6\lib\pyximport',
#        r'.\build\exe.win-amd64-3.6\lib\requests',
        r'.\build\exe.win-amd64-3.6\lib\scipy',
#        r'.\build\exe.win-amd64-3.6\lib\setuptools',
#        r'.\build\exe.win-amd64-3.6\lib\sphinx',
#        r'.\build\exe.win-amd64-3.6\lib\sphinxcontrib',
#        r'.\build\exe.win-amd64-3.6\lib\sqlalchemy',
#        r'.\build\exe.win-amd64-3.6\lib\sqlite3',
#        r'.\build\exe.win-amd64-3.6\lib\tables',
#        r'.\build\exe.win-amd64-3.6\lib\testpath',
        r'.\build\exe.win-amd64-3.6\lib\tkinter',
#        r'.\build\exe.win-amd64-3.6\lib\tornado',
#        r'.\build\exe.win-amd64-3.6\lib\traitlets',
        r'.\build\exe.win-amd64-3.6\lib\unittest',
        r'.\build\exe.win-amd64-3.6\lib\urllib',
        r'.\build\exe.win-amd64-3.6\lib\urllib3',
#        r'.\build\exe.win-amd64-3.6\lib\wcwidth',
#        r'.\build\exe.win-amd64-3.6\lib\webencodings',
#        r'.\build\exe.win-amd64-3.6\lib\win32com',
        r'.\build\exe.win-amd64-3.6\lib\win_unicode_console',
        r'.\build\exe.win-amd64-3.6\lib\xlrd',
        r'.\build\exe.win-amd64-3.6\lib\xlsxwriter',
        r'.\build\exe.win-amd64-3.6\lib\xlwt',
        r'.\build\exe.win-amd64-3.6\lib\xml',
        r'.\build\exe.win-amd64-3.6\lib\xmlrpc',
#        r'.\build\exe.win-amd64-3.6\lib\zmq',
#        r'.\build\exe.win-amd64-3.6\lib\_pytest',
        ]

# Remove all the folders NOT in the list keep_folders
root = r'.\build\exe.win-amd64-3.6\lib'
for folder in os.listdir(root):
    folder = os.path.join(root, folder)
    if os.path.isdir(folder):
        if folder not in keep_folders:
            shutil.rmtree(folder, ignore_errors=True)

# Remove some more specific folders:
remove_folders = [
        r'.\build\exe.win-amd64-3.6\mpl-data',
        r'.\build\exe.win-amd64-3.6\tk\demos',
        r'.\build\exe.win-amd64-3.6\lib\pandas\tests',
        ]
for folder in remove_folders:
    shutil.rmtree(folder, ignore_errors=True)
