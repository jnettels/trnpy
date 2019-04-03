# -*- coding: utf-8 -*-
'''
@author: Joris Nettelstroth

TRNpy: Parallelized TRNSYS simulation with Python
=================================================
Simulate TRNSYS deck files in serial or parallel and use parametric tables to
perform simulations for different sets of parameters. TRNpy helps to automate
these and similar operations by providing functions to manipulate deck files
and run TRNSYS simulations from a programmatic level.


Usage
=====
TRNpy can be used as a standalone application or imported into other Python
scripts.

Standalone TRNpy
----------------
TRNpy can be compiled into a Windows .exe file with the script ``setup.py``
and the following tips are valid for both ``trnpy.py`` and ``trnpy.exe``.

* By double-clicking the program, the main() function of this script is
  executed. It performs the most common tasks possible with this application:

  * The first file dialog allows to choose one or multiple deck files to
    simulate.
  * The second file dialog allows to choose a parametric table, which can be
    an Excel or a csv file. The first row must contain names of parameters
    you want to change. The following rows must contain the values of those
    parameters for each simulation you want to perform. TRNpy will make
    the substitutions in the given deck file and perform all the simulations.
  * You can cancel the second dialog to perform regular simulations.
  * The parametric table could look like this, to modify two parameters
    defined in TRNSYS equations:

    ===========  ===========
    Parameter_1  Parameter_2
    ===========  ===========
    100          0
    100          1
    200          0
    200          1
    ===========  ===========

* Running the program from a command line gives you more options, because
  you can use the built-in argument parser:

  * Type ``python trnsys.py --help`` or ``trnsys.exe --help`` to see the help
    message and an explanation of the available arguments.
  * This allows e.g. to enable parallel computing, hide the TRNSYS windows,
    suppress the parametric table file dialog, define the folder where
    parallel simulations are performed, and some more.
  * Example command:

    .. code::

        trnpy.exe --parallel --copy_files --n_cores 4 --table disabled

* Creating a shortcut to the executable is another practical approach:

  * Arguments can be appended to the path in the ``Target`` field
  * Changing the field ``Start in`` to e.g. ``C:\Trnsys17\Work`` will always
    open the file dialogs in that folder

Module Import
-------------
Import this script as a module into your own Python script. There you can
initialize objects of the ``DCK_processor()`` and ``TRNExe()`` classes and use
their functions. The first can create ``dck`` objects from regular TRNSYS
input (deck) files and manipulate them, the latter can run simulations with
the given ``dck`` objects.
This also gives you the option to perform post-processing tasks with
the simulation results (something that cannot be automated in the standalone
version).


Installation
============

TRNpy
-----
Just save the TRNpy application anywhere. As explained, it makes sense to
create shortcuts to the executable from your TRNSYS work folders.

Python
------
You do not need a Python installation for using ``trnpy.exe``.

If you want to use Python but do not yet have it installed, the easiest way to
do that is by downloading and installing **Anaconda** from here:
https://www.anaconda.com/download/
It's a package manager that distributes Python with data science packages.

During installation, please allow to add variables to ``$PATH`` (or do that
manually afterwards.) This allows Python to be started via command line from
every directory, which is very useful.


Support
=======
For questions and help, contact Joris Nettelstroth.
If you would like to request or contribute changes, do the same.

'''

import argparse
import logging
import os
import multiprocessing
from tkinter import Tk, filedialog

from trnpy.core import TRNExe, DCK_processor

# Define the logging function
logger = logging.getLogger(__name__)


def file_dialog_dck(initialdir=os.getcwd()):
    '''This function presents a file dialog for one or more TRNSYS deck files.

    Args:
        None

    Return:
        paths (List): List of file paths
    '''
    title = 'Please choose a TRNSYS Input File (*.dck)'
    logger.info(title)
    root = Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(
                initialdir=initialdir, title=title,
                filetypes=(('TRNSYS Input File', '*.dck'),)
                )
    files = list(files)
    if files == []:
        paths = None
    else:
        paths = [os.path.abspath(dck_file) for dck_file in files]
    return paths


def file_dialog_parametrics(initialdir=os.getcwd()):
    '''This function presents a file dialog for a parametric table file.

    Args:
        None

    Return:
        path (str): File path
    '''
    title = 'Choose a parametric table, or cancel to perform '\
            'a regular simulation'
    logger.info(title)
    root = Tk()
    root.withdraw()
    file = filedialog.askopenfilename(
                initialdir=initialdir, title=title,
                filetypes=(('Excel File', '*.xlsx'),
                           ('CSV File', '*.csv *.txt *.dat'))
                )
    if file == '':
        path = None
    else:
        path = os.path.abspath(file)
    return path


def str2bool(v):
    '''Convert a string to a boolean value. This is used in argparse to allow
    the input of boolean values from the command line.
    '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def perform_config(trnexe, dck_proc):
    '''Many configuration settings can be accessed by the user with the
    help of a YAML config file. If no config file exists, it is created
    with the current settings. If it exists, the settings are loaded.

    Args:
        TRNExe (TRNExe object): An instance of the TRNExe class

        dck_proc (DCK_processor object): An instance of the DCK_processor class

    Returns:
        None
    '''
    import yaml
    import sys

    if getattr(sys, 'frozen', False):  # Application is frozen (Windows .exe)
        folder = os.path.dirname(sys.executable)
    else:  # The application is not frozen (regular Python use)
        folder = os.path.dirname(__file__)

    config_file = os.path.join(folder, 'trnpy_config.yaml')

    config = {'TRNExe': trnexe.__dict__,
              'DCK_processor': dck_proc.__dict__,
              'logger': {'level': logger.level},
              }
    config['1 Info'] = \
        ['This is a YAML configuration file for the program TRNpy',
         'These settings are used as defaults for the argument parser',
         'You can use "#" to comment out lines in this file',
         'To restore the original config, just delete this file and '
         'restart the program TRNpy',
         'Run TRNpy with command --help to view the description of the '
         'arguments']

    if not os.path.exists(config_file):
        yaml.dump(config, open(config_file, 'w'), default_flow_style=False)
    else:
        try:
            config = yaml.load(open(config_file, 'r'))
            for key, value in config['TRNExe'].items():
                trnexe.__dict__[key] = value
            for key, value in config['DCK_processor'].items():
                dck_proc.__dict__[key] = value
            for key, value in config['logger'].items():
                logger.__dict__[key] = value

        except Exception as ex:
            logging.error(str(ex))


def run_OptionParser(TRNExe, dck_proc):
    '''Define and run the option parser. Set the user input and return the list
    of decks. Needs TRNExe and dck_proc to get and set the option values.

    Args:
        TRNExe (TRNExe object): An instance of the TRNExe class

        dck_proc (DCK_processor object): An instance of the DCK_processor class

    Returns:
        dck_list (list): A list of dck objects

    '''
    description = 'TRNpy: Parallelized TRNSYS simulation with Python. '\
        'Simulate all the TRNSYS deck files specified as DCK '\
        'arguments in serial or parallel. Specify a '\
        'PARAMETRIC_TABLE file to perform simulations for different'\
        ' sets of parameters. See documentation for further help.'
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)

#    parser.add_argument('--version', action='version', version='%(prog)s 0.3')

    group1 = parser.add_argument_group('Basic options', 'Use the ' +
                                       'following options to define how to ' +
                                       'run this program.')

    group1.add_argument('-d', '--deck', dest='dck', help='One or more paths ' +
                        'to TRNSYS input files (*.dck). If not specified, ' +
                        'a file dialog opens instead', type=str, nargs='+',
                        default=[])

    group1.add_argument('--hidden', action='store_true',
                        dest='mode_trnsys_hidden',
                        help='Hide all TRNSYS windows',
                        default=TRNExe.mode_trnsys_hidden)

    group1.add_argument('-p', '--parallel', action='store_true',
                        dest='mode_exec_parallel',
                        help='Run simulations in parallel',
                        default=TRNExe.mode_exec_parallel)

    group1.add_argument('-c', '--copy_files', action='store_true',
                        dest='copy_files',
                        help='Copy simulation files to a new folder within ' +
                        'SIM_FOLDER. This helps to prevent conflicts ' +
                        'between different simulations.',
                        default=False)

    group1.add_argument('-t', '--table', action='store', type=str,
                        dest='parametric_table',
                        help='Path to a PARAMETRIC_TABLE file with ' +
                        'replacements to be made in the given deck files. ' +
                        'If not specified, a file dialog opens instead. ' +
                        'Pass "disabled" to suppress the file dialog.',
                        default=None)

    group2 = parser.add_argument_group('Advanced options', 'These options ' +
                                       'do not need to be changed in most ' +
                                       'cases.')

    group2.add_argument('-l', '--log_level', action='store', dest='log_level',
                        help='LOG_LEVEL can be one of: debug, info, ' +
                        'warning, error or critical',
                        default=logging.getLevelName(
                                logger.getEffectiveLevel()))

    group2.add_argument('--sim_folder', action='store',
                        dest='sim_folder',
                        help='Folder where new simulations are created in, ' +
                        'if --copy_files is true or PARAMETRIC_TABLE is' +
                        ' given',
                        default=dck_proc.sim_folder)

    group2.add_argument('--regex_result_files', action='store',
                        dest='regex_result_files',
                        help='We need to separate the input files for your ' +
                        'deck(s) from the output/result files. Please enter ' +
                        ' a pattern that only occurs in the file paths of ' +
                        'result files, regular expressions are supported.',
                        default=dck_proc.regex_result_files)

    group2.add_argument('--path_TRNExe', action='store',
                        dest='path_TRNExe',
                        help='Path to the TRNExe.exe.',
                        default=TRNExe.path_TRNExe)

    group2.add_argument('--n_cores', action='store', type=int,
                        dest='n_cores',
                        help='Number of CPU cores to use for parallelization' +
                        '. "0" is for detection of total number minus one.',
                        default=TRNExe.n_cores)

    group2.add_argument('--check_vital_sign', action='store', type=str2bool,
                        dest='check_vital_sign',
                        help="""Determine whether or not a TRNSYS simulation is
                        "alive" by checking the CPU load. This allows to
                        automatically detect and quit simulations that
                        ended with an error. You need to disable this
                        feature if you want to be able to pause and resume
                        the live plotter during a simulation.""",
                        default=TRNExe.check_vital_sign)

    # Read the user input:
    args, unknown = parser.parse_known_args()
    args.dck += unknown  # any "unknown" arguments are also treated as decks

    # Save user input by overwriting the default values:
    TRNExe.path_TRNExe = args.path_TRNExe
    TRNExe.mode_trnsys_hidden = args.mode_trnsys_hidden
    TRNExe.mode_exec_parallel = args.mode_exec_parallel
    TRNExe.n_cores = args.n_cores
    TRNExe.check_vital_sign = args.check_vital_sign
    dck_proc.sim_folder = os.path.abspath(args.sim_folder)
    dck_proc.regex_result_files = args.regex_result_files

    # Set level of logging function
    logger.setLevel(level=args.log_level.upper())
    logging.getLogger('trnpy.core').setLevel(level=args.log_level.upper())
    logging.getLogger('trnpy.misc').setLevel(level=args.log_level.upper())

    if len(args.dck) == 0:
        dck_file_list = file_dialog_dck()
        if dck_file_list is None:
            logger.info('Empty selection. Show help and exit program...')
            parser.print_help()
            input('\nPress the enter key to exit.')
            raise SystemExit
    else:
        # Get list of deck files (and convert relative into absolute paths)
        dck_file_list = [os.path.abspath(dck_file) for dck_file in args.dck]

    logger.debug('List of dck files:')
    if logger.isEnabledFor(logging.DEBUG):
        for dck_file in dck_file_list:
            print(dck_file)

    parametric_table = args.parametric_table
    # All the user input is read in. Now the appropriate action is taken
    if parametric_table != 'disabled':
        if parametric_table is None:
            # Show a dialog window for file selection (no selection is 'None')
            parametric_table = file_dialog_parametrics()
        if parametric_table is not None:
            # A parametric table was given. Automate the default procedure
            dck_list = dck_proc.parametric_table_auto(parametric_table,
                                                      dck_file_list)
        else:
            parametric_table = 'disabled'

    if parametric_table == 'disabled':
        # Just a list of decks to simulate, without any modifications
        # Depending on --copy_files, simulate in the --sim_folder or in the
        # original folder
        dck_list = dck_proc.create_dcks_from_file_list(
                      dck_file_list,
                      update_dest=args.copy_files,
                      copy_files=args.copy_files)

    return dck_list  # Return a list of dck objects


if __name__ == "__main__":
    '''Main function
    This function is executed when the script is started directly with
    Python, not when it is loaded as a module.
    '''
    multiprocessing.freeze_support()  # Required on Windows

    # Define output format of logging function
    logging.basicConfig(format='%(asctime)-15s %(message)s')
    logger.setLevel(level='INFO')  # Set a default level for the logger

    try:
        trnexe = TRNExe()  # Create TRNExe object
        dck_proc = DCK_processor()  # Create DCK_processor object
        perform_config(trnexe, dck_proc)  # Save or load the config file
        dck_list = run_OptionParser(trnexe, dck_proc)  # Get user input
        dck_list = trnexe.run_TRNSYS_dck_list(dck_list)  # Perform simulations
        dck_proc.report_errors(dck_list)  # Show any simulation errors
    except Exception as ex:
        logger.exception(ex)

    input('\nPress the enter key to exit.')
