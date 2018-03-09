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
import re
import os
import shutil
import multiprocessing
import subprocess
import time
import pandas as pd
import psutil
import hashlib
import itertools
from tkinter import Tk, filedialog
from bokeh.command.bootstrap import main

# Default values that are used by multiple classes:
regex_result_files_def = r'Result|\.sum|\.pr.'


class TRNExe(object):
    '''The TRNExe class.
    The most prominent function a user will need is ``run_TRNSYS_dck_list()``,
    in order to perform the actual TRNSYS simulations with a list of ``dck``
    objects. All other functions are only used internally by that function.

    The behaviour of the TRNExe object (run in parallel, run hidden, number
    of CPU cores used) is controlled by the options given at initialization.
    '''

    def __init__(self,
                 path_TRNExe=r'C:\Trnsys17\Exe\TRNExe.exe',
                 mode_trnsys_hidden=False,
                 mode_exec_parallel=False,
                 n_cores=0
                 ):
        '''
        The optional argument n_cores allows control over the used CPU cores.
        By default, the total number of CPU cores minus one are used. This way
        your CPU is not under 100% load - and using all cores does not
        necessarily make total simulation time faster.

        Args:
            path_TRNExe (str): Path to the actual TRNSYS executable

            mode_trnsys_hidden (bool): Run simulations completely hidden?

            mode_exec_parallel (bool): Run simulations in parallel?

            n_cores (int, optional): Number of CPU cores to use in parallel

        Returns:
            None
        '''
        self.path_TRNExe = path_TRNExe
        self.mode_trnsys_hidden = mode_trnsys_hidden
        self.mode_exec_parallel = mode_exec_parallel
        self.n_cores = n_cores

    def run_TRNSYS_dck(self, dck):
        '''Run a TRNSYS simulation with the given deck dck_file.
        '''
        if not os.path.exists(self.path_TRNExe):
            raise FileNotFoundError('TRNExe.exe not found: '+self.path_TRNExe)

        if self.mode_trnsys_hidden:
            mode_trnsys = '/h'  # hidden mode
        else:
            mode_trnsys = '/n'  # batch mode

        proc = subprocess.Popen([self.path_TRNExe, dck.file_path_dest,
                                 mode_trnsys])

        logging.debug('TRNSYS started with PID ' + str(proc.pid) +
                      ' ('+dck.file_path_dest+')')

        # Wait until the process is finished
        while proc.poll() is None:
            # While we wait, we check if TRNSYS is still running
            if self.TRNExe_is_alive(proc.pid) is False:
                logging.debug('TRNSYS is inactive and may have encountered '
                              'an error, killing the process in 10 seconds: ' +
                              dck.file_path_dest)
                time.sleep(10)
                proc.terminate()
                dck.error_msg_list.append('Low CPU load - timeout')
                dck.success = False

        # Check for error messages in the log and store them in the deck object
        if self.check_log_for_errors(dck) is False or dck.success is False:
            dck.success = False
            logging.debug('Finished PID ' + str(proc.pid) + ' with errors')
        else:
            logging.debug('Finished PID ' + str(proc.pid))
            dck.success = True
        # Return the deck object
        return dck

    def check_log_for_errors(self, dck):
        '''Stores errors in error_msg_list. Returns False if errors are found.
        Function tries to open the log file of the given simulation and
        searches for error messages with a regular expression.

        Args:
            dck (dck object): Simulation to check for errors

        Returns:
            True/False (bool): Did simulation finish successfully?
        '''
        logfile_path = os.path.splitext(dck.file_path_dest)[0] + '.log'
        try:
            with open(logfile_path, 'r') as f:
                logfile = f.read()
        except FileNotFoundError:
            dck.error_msg_list.append('No logfile found')
            return False

        re_msg = r'Fatal Error.*\n.*\n.*\n.*Message.*\:\ (?P<msg>.*\n.*)'
        match_list = re.findall(re_msg, logfile)
        if match_list:  # Matches of the regular expression were found
            for msg in match_list:
                dck.error_msg_list.append(msg)  # Store error messages
            return False
        else:  # No error messages found
            return True

    def TRNExe_is_alive(self, pid, interval_1=0.3, interval_2=9.7):
        '''Check whether or not a particular TRNExe.exe process is alive.
        This status is guessed by measuring its CPU load over the given time
        intervals. The first measurement has to be short, because it blocks
        the thread. If it indicates a low CPU load, the second (longer)
        measurement is performed to be sure.

        Args:
            pid (int): Process ID number

            interval_1 (float): First (short) interval in seconds

            interval_2 (float): Second (longer) interval in seconds

        Returns:
            True/False (bool): Status information
        '''
        try:
            # The process might have finished by the time we get to check
            # its status. This would cause an error.
            p = psutil.Process(pid)
            if p.cpu_percent(interval=interval_1) < 0.1:
                if p.cpu_percent(interval=interval_2) < 0.1:
                    # print(p.memory_info())
                    return False
        except Exception:
            pass

        return True

    def run_TRNSYS_dck_list(self, dck_list):
        '''Run one TRNSYS simulation of each deck file in the dck_list.
        This is a wrapper around run_TRNSYS_dck() that allows execution in
        serial and in parallel.

        Args:
            dck_list (list): List of DCK objects to work on

        Returns:
            returned_dck_list (list): List of DCK objects worked on
        '''
        if len(dck_list) == 0:
            raise ValueError('The list of decks "dck_list" must not be empty.')

        start_time = time.time()
        if not self.mode_exec_parallel:
            returned_dck_list = []
            for dck in dck_list:
                returned_dck = self.run_TRNSYS_dck(dck)
                returned_dck_list.append(returned_dck)

        if self.mode_exec_parallel:
            n_cores = self.n_cores
            if n_cores == 0:
                n_cores = min(multiprocessing.cpu_count() - 1, len(dck_list))

            logging.info('Parallel processing of ' + str(len(dck_list)) +
                         ' jobs on ' + str(n_cores) + ' cores')
            pool = multiprocessing.Pool(n_cores)

            '''For short lists, imap seemed the fastest option.
            With imap, the result is a consumable iterator'''
#            map_result = pool.imap(self.run_TRNSYS_dck, dck_list)
            '''With map_async, the results are available immediately'''
            map_result = pool.map_async(self.run_TRNSYS_dck, dck_list)

            pool.close()  # No more processes can be launched
            self.print_progress(map_result, start_time)
            pool.join()  # Workers are removed

            returned_dck_list = map_result.get()
            # With imap, the iterator must be turned into a list:
#            returned_dck_list = list(map_result)
#            print(returned_dck_list)

        script_time = pd.to_timedelta(time.time() - start_time, unit='s')
        script_time = str(script_time).split('.')[0]
        logging.info('Finished all simulations in time: %s' % (script_time))

        return returned_dck_list

    def print_progress(self, map_result, start_time):
        '''Prints info about the multiprocessing progress to the screen.
        Measures and prints the percentage of simulations done and an
        estimation of the remaining time.
        The parameter 'map_result' must be the return of a pool.map_async()
        function. Please be aware that this mapping splits the work list into
        'chunks'. 'map_result._number_left' only updates once one chunk has
        been completed. Therefore the progress percentage does not necessarily
        update after every finished TRNSYS simulation.

        Args:
            map_result (map_async object): Return of 'pool.map_async'

            start_time (time): Simulation start time

        Returns:
            None
        '''
        total = map_result._number_left
        remaining_last = total
        while map_result.ready() is False:  # Repeat this until finished
            remaining = map_result._number_left
            fraction = (total - remaining)/total
            t_elapsed = pd.to_timedelta(time.time() - start_time, unit='s')

            if total - remaining != 0:
                # The remaining time estimation is only done after at least
                # one job has finished
                if remaining_last != remaining:
                    # Update the 'total' estimation, when new jobs have
                    # finished
                    t_total = t_elapsed/(total - remaining)*total
                    remaining_last = remaining
                elif t_total < t_elapsed:
                    t_total = t_elapsed
                t_remain = t_total - t_elapsed
                # Format the text message
                text = '{:5.1f}% done. Time elapsed: {}, remaining: {}'\
                       .format(fraction*100,
                               str(t_elapsed).split('.')[0],
                               str(t_remain).split('.')[0])
            else:
                # Until the first job has finished, do this:
                text = '{:5.1f}% done. Time elapsed: {}'\
                       .format(fraction*100,
                               str(t_elapsed).split('.')[0])
            # Do the print
            print(text, end='\r')
            time.sleep(1.0)  # Sleep a certain number of seconds
        print('\x1b[2K', end='\r')  # Clear last output line
        return


class DCK(object):
    '''Deck class.
    Holds all the information about a deck file, including the content of the
    file itself. This allows manipulating the content, before saving the
    actual file to the old or a new location for simulation.
    These locations are stored, as well as the applied manipulations.
    After a simulation, potential errors can be stored in the object, too.
    '''
    def __init__(self, file_path, regex_result_files=regex_result_files_def):
        self.file_path_orig = file_path
        self.file_path_dest = file_path
        self.file_name = os.path.splitext(os.path.basename(file_path))[0]
        self.error_msg_list = []
        self.success = None
        self.hash = None
        self.replace_dict = dict()
        self.regex_dict = dict()
        self.regex_result_files = regex_result_files
        self.dck_text = ''

        # Perform the following functions to initialize some more values
        self.load_dck_text()
        self.find_assigned_files()

    def load_dck_text(self):
        '''Here we store the complete text of the deck file as a property of
        the deck object.

        .. note::
            This may or may not prove to consume too much memory.

        Args:
            None

        Returns:
            None
        '''
        if os.path.splitext(self.file_path_orig)[1] != '.dck':
            msg = self.file_path_orig+' has the wrong file type, must be .dck'
            raise ValueError(msg)
        else:
            # Deck files are not saved as UTF-8 by TRNSYS, so reading them
            # can cause problems. The following helps to prevent issues:
            with open(self.file_path_orig, 'r', encoding="WINDOWS-1252") as f:
                self.dck_text = f.read()

    def find_assigned_files(self):
        '''Find all file paths that appear after an "ASSIGN" key in the
        deck. The files are sorted into input and output (=result) files by
        applying regex_result_files.
        The results are stored in the lists assigned_files and result_files.

        Args:
            None

        Returns:
            None
        '''
        self.assigned_files = []
        self.result_files = []
        self.assigned_files = re.findall(r'ASSIGN \"(.*)\"', self.dck_text)
        for file in self.assigned_files.copy():
            if re.search(self.regex_result_files, file):
                self.assigned_files.remove(file)
                self.result_files.append(file)

        if len(self.result_files) == 0:
            logging.warning('No result files were identified among the '
                            'assigned files in the deck '+self.file_name+'. '
                            'This may cause issues. Is this regular expression'
                            ' correct? "'+self.regex_result_files+'"')


class DCK_processor(object):
    '''Deck processor class.
    Create ``dck`` objects from regular TRNSYS input (deck) files and
    manipulate them. An example workflow could be:
        * For parameter variations:

            * ``parametric_table_read()``
            * ``get_parametric_dck_list()``
        * or for a simple list of deck files with manual replacements:

            * ``create_dcks_from_file_list()``
            * ``add_replacements_value_of_key()``
        * ``rewrite_dcks()``
        * ``copy_assigned_files()``

    Then ``run_TRNSYS_dck_list()`` of a ``TRNExe`` object can be used to
    simulate a ``dck_list``.

    Additionally, post-processing functions are available which can be
    used e.g. in the following order:

        * ``report_errors()``
        * ``results_collect()``
        * ``results_create_index()``
        * ``results_slice_time()``
        * ``results_resample()``

    This allows to collect the simulation results and store them as DataFrames,
    create a proper Pandas multiindex from the results of parametric runs,
    slice to select specific time intervals and / or resample the data to
    new frequencies, e.g. from hours to months.
    '''
    def __init__(self, root_folder=r'C:\Trnsys17\Work\batch',
                 regex_result_files=regex_result_files_def):
        self.root_folder = root_folder
        self.regex_result_files = regex_result_files

    def parametric_table_auto(self, parametric_table, dck_file_list):
        '''Convenient automated parametric table function.
        A parametric table was given. Therefore we do the standard procedure
        of creating a ``dck`` object list from the parameters. We add those
        lists for all given files.

        Args:
            parametric_table (DataFrame): Pandas DataFrame

            dck_file_list (list): List of file paths

        Returns:
            dck_list (list): List of dck objects
        '''
        dck_list = []
        for dck_file in dck_file_list:
            dck_list += self.get_parametric_dck_list(parametric_table,
                                                     dck_file)
        self.rewrite_dcks(dck_list)
        self.copy_assigned_files(dck_list)
        return dck_list

    def parametric_table_read(self, param_table_file):
        '''Reads a parametric table from a given file and return it as a
        DataFrame. Uses ``read_filetypes()`` to read the file.

        Args:
            param_table_file (str): Path to a file

        Returns:
            parametric_table (DataFrame): Pandas DataFrame
        '''
        parametric_table = self.read_filetypes(param_table_file)

        logging.info(param_table_file+':')
        if logging.getLogger().isEnabledFor(logging.INFO):
            print(parametric_table)

        return parametric_table

    def parametric_table_combine(self, parameters):
        '''Produces a parametric table from all combinations of individual
        values of the given parameters.

        Args:
            parameters (dict): Dictionary with parameters and their values

        Returns:
            parametric_table (DataFrame): Pandas DataFrame
        '''
        flat = [[(k, v) for v in vs] for k, vs in parameters.items()]
        combis = [dict(items) for items in itertools.product(*flat)]
        parametric_table = pd.DataFrame.from_dict(combis)

        logging.info('Parametric table from combinations:')
        if logging.getLogger().isEnabledFor(logging.INFO):
            print(parametric_table)

        return parametric_table

    def gen_row_hash(self):
        '''Unused'''
        string = 'a'
        hash = hashlib.sha1(string).hexdigest()
        return hash

    def read_filetypes(self, filepath, **kwargs):
        '''Read any file type with stored data and return the Pandas DataFrame.
        Wrapper around Pandas' read_excel() and read_csv().

        Please note: With 'kwargs', you can pass any (named) parameter down
        to the Pandas functions. The TRNSYS printer adds some useless rows
        at the top and bottom of the file? No problem, just define 'skiprows'
        and 'skipfooter'. For all options, see:
        http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
        '''
        filetype = os.path.splitext(os.path.basename(filepath))[1]
        if filetype in ['.xlsx', '.xls']:
            # Excel can be read automatically
            df = pd.read_excel(filepath, **kwargs)  # Pandas function
        elif filetype in ['.csv']:
            # Standard format: Here we guess everything. May or may not work
            df = pd.read_csv(filepath,
                             sep=None, engine='python',  # Guess separator
                             parse_dates=[0],  # Try to parse column 0 as date
                             infer_datetime_format=True,
                             **kwargs)
        elif filetype in ['.out']:
            # Standard format for TRNSYS: Separator is whitespace
            df = pd.read_csv(filepath,
                             delim_whitespace=True,
                             encoding='WINDOWS-1252',  # TRNSYS encoding
                             **kwargs)
        elif filetype in ['.dat', '.txt']:
            logging.warning('Unsupported file extension: ' + filetype +
                            '. Trying to read it like a csv file.')
            df = pd.read_csv(filepath, **kwargs)
        else:
            raise NotImplementedError('Unsupported file extension: "' +
                                      filetype + '" in file ' + filepath)

        return df

    def get_parametric_dck_list(self, parametric_table, dck_file):
        if isinstance(parametric_table, str):
            # Default is to hand over a parametric_table DataFrame. For
            # convenience, a file path is accepted and read into a DataFrame
            parametric_table = self.parametric_table_read(parametric_table)

        # Start building the list of deck objects
        dck_list = []
        for hash in parametric_table.index.values:
            # For each row in the table, one deck object is created and
            # updated with the correct information.
            dck = DCK(dck_file, regex_result_files=self.regex_result_files)
            dck.hash = str(hash)
            dck.file_path_dest = os.path.join(self.root_folder,
                                              dck.file_name,
                                              dck.hash,
                                              dck.file_name+'.dck')

            # Store the replacements found in the deck object
            for col in parametric_table.columns:
                dck.replace_dict[col] = parametric_table.loc[hash][col]
            # Convert the replacements into regular expressions
            self.add_replacements_value_of_key(dck.replace_dict, dck)
            # Done. Add the deck to the list.
            dck_list.append(dck)
        return dck_list

    def add_replacements(self, replace_dict_new, dck):
        '''Add new entries to the existing replace_dict.
        Basic use is to add strings, but also accepts regular expressions.
        These allow you to replace any kind of information.

        Args:
            replace_dict_new (dict): Dictionary of 'str_old: string_new' pairs

        Returns:
            None
        '''
        for re_find, re_replace in replace_dict_new.items():
            dck.regex_dict[re_find] = re_replace

    def add_replacements_value_of_key(self, replace_dict_new, dck):
        '''Add new entries to the existing replace_dict.
        Creates the required regular expression to replace the 'value' of the
        'key' in the deck file. Then calls add_replacements().

        Args:
            replace_dict_new (dict): Dictionary of 'key: value' pairs

        Returns:
            None

        Example:
            replace_dict_new = {'A_Koll': 550, 'plot_on_off': -1}
        '''
        for key, value in replace_dict_new.items():
            # Find key and previous value, possibly separated by '='
            re_find = r'(?P<key>\b'+key+'\s?=\s?)(?P<value>.*)'
#            re_find = r'(?P<key>\b'+key+'\s=\s)(?P<value>\W*\d*\W?\d*\n)'
            # Replace match with key (capture group) plus the new value
            re_replace = r'\g<key>'+str(value)
#            re_replace = r'\g<key>'+str(value)+'\n'
            self.add_replacements({re_find: re_replace}, dck)

    def disable_plotters(self, dck):
            '''Disable the plotters by setting their parameter 9 to '-1'.
            Calls add_replacements() with the required regular expressions.
            '''
            re_find = r'\S+(?P<old_text>\s*! 9 Shut off Online )'
            re_replace = r''+str(-1)+'\g<old_text>'
            self.add_replacements({re_find: re_replace}, dck)

    def reset_replacements(self, dck):
        '''Reset the replace_dict to make it empty.
        '''
        dck.replace_dict = dict()

    def create_dcks_from_file_list(self, dck_file_list, update_dest=False):
        '''Takes a list of file paths and creates deck objects for each one.
        If the optional argument 'update_dest' is True, the destination path
        is updated and based on the 'root_folder'.
        The default False means simulating the deck in the original folder.

        Please make sure that the dck_file_list contains actual paths.
        If the deck file is in the same folder as the script, the path has
        to be "./TRNSYS_input_file.dck".
        '''
        dck_list = [DCK(dck_file, regex_result_files=self.regex_result_files)
                    for dck_file in dck_file_list]
        if update_dest:
            for dck in dck_list:
                dck.file_path_dest = os.path.join(self.root_folder,
                                                  dck.file_name,
                                                  dck.file_name+'.dck')
        return dck_list

    def rewrite_dcks(self, dck_list):
        '''Perform the replacements in self.replace_dict in the deck files.
        You have to use add_replacements(), add_replacements_value_of_key()
        or disable_plotters() before, to fill the replace_dict.

        Args:
            dck_list (list): List of paths to deck files to work on

        Returns:
            dck_list (list): List of paths
        '''
        # Process the deck file(s)
        for dck in dck_list:
            # Perform the replacements:
            for re_find, re_replace in dck.regex_dict.items():
                dck.dck_text, number_of_subs_made = re.subn(re_find,
                                                            re_replace,
                                                            dck.dck_text)
                if number_of_subs_made == 0:
                    logging.warning('Warning: Replacement not successful, ' +
                                    'because regex was not found: '+re_find)

        return

    def copy_assigned_files(self, dck_list):
        '''All external files that a TRNSYS deck depends on are stored with
        the ASSIGN parameter. We make a list of those files and copy them
        to the required location. (Weather data, load profiles, etc.)
        '''
        for dck in dck_list:
            # Copy the assigned files
            source_folder = os.path.dirname(dck.file_path_orig)
            destination_folder = os.path.dirname(dck.file_path_dest)
            for file in dck.assigned_files:
                source_file = os.path.join(source_folder, file)
                destination_file = os.path.join(destination_folder, file)
#                logging.debug('source      = '+source_file)
#                logging.debug('destination = '+destination_file)
                if not os.path.exists(os.path.dirname(destination_file)):
                    os.makedirs(os.path.dirname(destination_file))

                if not source_file == destination_file:
                    try:
                        shutil.copy2(source_file, destination_file)
                    except Exception as ex:
                        logging.debug('Error in ' + dck.file_name)
                        raise
                else:
                    logging.debug(dck.file_name + ': Copy source and ' +
                                  'destination are equal for file:')
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        print(source_file)

            # For result files, we must create the folders
            for file in dck.result_files:
                destination_file = os.path.join(destination_folder, file)
                if not os.path.exists(os.path.dirname(destination_file)):
                    os.makedirs(os.path.dirname(destination_file))

            # Store the deck files:
            if not os.path.exists(os.path.dirname(dck.file_path_dest)):
                os.makedirs(os.path.dirname(dck.file_path_dest))

            with open(os.path.join(dck.file_path_dest), "w") as f:
                    f.write(dck.dck_text)

        # Print the list of the created & copied parametric deck files
        logging.debug('List of copied dck files:')
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            for dck in dck_list:
                print(dck.file_path_dest)

        return

    def report_errors(self, dck_list):
        '''Print all the errors stored in the ``dck`` objects of the given
        dck_list.

        Args:
            dck_list (list): List of ``dck`` objects

        Returns:
            None
        '''
        for dck in dck_list:
            if dck.success is False:
                print('Errors in ' + dck.file_path_dest)
                for i, error_msg in enumerate(dck.error_msg_list):
                    print('  '+str(i)+': '+error_msg)
                print('')  # Finish with an empty line
        return

    def results_collect(self, dck_list, read_file_function, create_index=True):
        '''Collect the results of the simulations. Our goal is to combine
        the result files of the parametric runs into DataFrames. The DataFrames
        contain the raw data plus columns for each of the replacements made
        in the deck files. This allows you to identify each parametric run.

        The assumption is that all simulations produced the same set of result
        files. The dict 'result_data' will have one DataFrame for each file
        name. In order for this to work, the TRNSYS out files have to be read
        in and converted into Pandas DataFrames. And different TRNSYS printer
        types format their output in different ways. Therefore there can be
        no fully automated reading of all TRNSYS results.

        Instead, you have to provide the function that manages reading the
        files ('read_file_function'). In some cases, it can look like this:

        .. code:: python

            def read_file_function(result_file_path):
                return dck_proc.read_filetypes(result_file_path)

        .. note::
            This means that you can utilize the existing read_filetypes(),
            which can already handle different file types. Please see the docs
            for read_filetypes() for info about passing over additional
            arguments to customize it to your needs.

        Args:
            dck_list (list): A list of DCK objects

            read_file_function (func): A function that takes one argument (a
            file path) and returns a Pandas DataFrame.

            create_index (bool, optional): Move time and parameter columns to
            the index of the DataFrame. Default: True

        Returns:
            result_data (dict): A dictionary with one DataFrame for each file
        '''
        result_data = dict()
        for dck in dck_list:
            for result_file in dck.result_files:
                # Construct the full path to the result file
                result_file_path = os.path.join(os.path.dirname(
                                                dck.file_path_dest),
                                                result_file)
                # Use the provided function to read the file
                try:
                    df_new = read_file_function(result_file_path)

                    # Add the 'hash' and all the key, value pairs to DataFrame
                    df_new['hash'] = [dck.hash]*len(df_new)
                    for key, value in dck.replace_dict.items():
                        df_new[key] = [value]*len(df_new)

                    # Add the DataFrame to the dict of result files
                    if result_file in result_data.keys():
                        df_old = result_data[result_file]
                    else:
                        df_old = pd.DataFrame()
                    # Append the old and new df, with a new index.
                    df = pd.concat([df_old, df_new], ignore_index=True)
                    # Add it to the dict
                    result_data[result_file] = df

                except Exception as ex:
                    logging.error('Error when trying to read result file "' +
                                  result_file + '": ' + str(ex))

        logging.info('Collected result files:')
        if logging.getLogger().isEnabledFor(logging.INFO):
            for file in result_data.keys():
                print(file)
        return result_data

    def results_create_index(self, result_data, replace_dict, origin):
        '''Put the time and parameter columns into the index of the DataFrame.
        The function expects the return of results_collect(). This typically
        creates a multi-index and is arguably how DataFrames are supposed to
        be handled.

        Args:
            result_data (dict): The return of the function results_collect()

            replace_dict (dict): Is required to identify the parameter columns

            origin (str): Start date, e.g. '2003-01-01'. TRNSYS does not care
            for years or days, but to get a pretty DataFrame we use datetime
            for the time column, which needs a fully defined date.

        Returns:
            result_data (dict): A dictionary with one DataFrame for each file
        '''
        for key, df in result_data.items():
            if 'TIME' in df.columns:
                t_col = 'TIME'
            elif 'Time' in df.columns:
                t_col = 'Time'
            else:
                continue
            # Convert TIME column to float and then to datetime
            df[t_col] = [float(string) for string in df[t_col]]
            df[t_col] = pd.to_datetime(df[t_col], unit='h',
                                       origin=pd.Timestamp(origin))

            # Create a list and use that as the new index columns
            idx_cols = list(replace_dict.keys()) + [t_col]
            df.set_index(keys=idx_cols, inplace=True)
            df.sort_index(inplace=True)

        return result_data

    def results_slice_time(self, df, start, end):
        '''Slice the time index from ``"start"`` to ``"end"``, while keeping
        the other index columns intact. Expects the time column to be at last
        position in the multi-index.

        Args:
            df (DataFrame): Pandas DataFrame to slice

            start (str): Start date/time for slicing interval

            end (str): End date/time for slicing interval

        Returns:
            df_new (DataFrame): Sliced DataFrame
        '''
        n_index_cols = len(df.index.names)  # Number of index columns
        if n_index_cols == 1:
            df_new = df.loc[start:end]  # Slice regular index
        else:
            slice_list = [slice(None)]*(n_index_cols-1) + [slice(start, end)]
            slices = tuple(slice_list)  # Convert list to tuple
            df_new = df.loc[slices, :]  # Slice multiindex
        return df_new

    def results_resample(self, df, freq, regex_sum=r'Q_|E_',
                         regex_mean=r'T_|M_', **kwargs):
        '''Resample a multi-indexed DataFrame to a new frequency.
        Expects the time column to be at last position in the multi-index.
        Columns not matched by the regular expressions will be dropped with
        an info message.

        Args:
            df (DataFrame): Pandas DataFrame to resample

            freq (str): Pandas frequency, e.g. ``"D"``, ``"W"``, ``"M"``
            or ``"Y"``. For a full list, see the Pandas documentation at
            http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

            regex_sum (str): Regular expression that matches all names of
            columns that should be summed up (e.g. energy values)

            regex_mean (str): Regular expression that matches all names of
            columns that should use the mean (e.g. temperatures)

        kwargs:
            Additional keyword arguments, can be used to pass ``"closed"``
            and/or ``"label"`` (each with the values ``"left"`` or ``"right"``)
            to the ``"resample()"`` function

        Returns:
            df_new (DataFrame): Resampled DataFrame
        '''
        # Apply the regular expressions to get two lists of columns
        cols_sum = []
        cols_mean = []
        cols_found = []
        for column in df.columns:
            if re.search(regex_sum, column):
                cols_sum.append(column)
                cols_found.append(column)
            elif re.search(regex_mean, column):
                cols_mean.append(column)
                cols_found.append(column)
            else:
                logging.info('Column "'+column+'" did not match the regular '
                             'expressions and will not be resampled')

        if len(df.index.names) == 1:
            # If there is only the time column in the index, the resampling
            # is straight forward
            sum_df = df[cols_sum].resample(freq, **kwargs).sum()
            mean_df = df[cols_mean].resample(freq, **kwargs).mean()

        else:
            # Perform the resampling while preserving the multiindex,
            # achieved with the groupby function
            level_values = df.index.get_level_values
            levels = range(len(df.index.names))[:-1]  # All columns except time

            sum_df = (df[cols_sum].groupby([level_values(i) for i in levels]
                      + [pd.Grouper(freq=freq, **kwargs, level=-1)]).sum())
            mean_df = (df[cols_mean].groupby([level_values(i) for i in levels]
                       + [pd.Grouper(freq=freq, **kwargs, level=-1)]).mean())
        # Recombine the two DataFrames into one
        df_new = pd.concat([sum_df, mean_df], axis=1)
        df_new = df_new[cols_found]  # Sort columns in their original order
        return df_new

    def DataExplorer_mark_index(self, df):
        '''Put '!' in front of index column names, to mark them as
        classifications. Is not applied to time columns.
        '''
        idx_cols_rename = []
        for name in df.index.names:
            if name not in ['TIME', 'Time', 'time']:
                idx_cols_rename.append('!'+name)
            else:
                idx_cols_rename.append(name)
        if len(idx_cols_rename) == 1:
            idx_cols_rename = idx_cols_rename[0]
        df.index = df.index.rename(idx_cols_rename)
        return df

    def DataExplorer_open(self, DatEx_df, data_name='TRNSYS Results', port=80,
                          bokeh_app=r'C:\Users\nettelstroth\Documents' +
                                    r'\07 Python\dataexplorer',
                          show=False):
        '''Open the given DataFrame in the DataExplorer application. TRNpy and
        DataExplorer are a great combination, because the values of parametric
        runs can be viewed and filtered as classes in the DataExplorer.
        '''
        # Mark index column names as classifications
        DatEx_df = self.DataExplorer_mark_index(DatEx_df)

        # Prepare settings:
        data_file = os.path.join(bokeh_app, 'upload', data_name + '.xlsx')
        logging.info('Saving file for DataExplorer... ')
        logging.info(data_file)
        print(DatEx_df.head())

        # Save this as a file that DataExplorer will load again
        DatEx_df.to_excel(data_file, merge_cells=False)

        logging.info('Starting DataExplorer...')
        try:
            call_list = ["bokeh", "serve", bokeh_app, "--port", str(port)]
            if show:
                call_list.append("--show")
            call_list += ["--args",
                          "--name", data_name,
                          "--file", data_file]
            # Call Bokeh app:
            main(call_list)
        except SystemExit:
            # Would produce ugly print (when port is already in use)
            pass


def file_dialog_dck(initialdir=os.getcwd()):
    '''This function presents a file dialog for one or more TRNSYS deck files.

    Args:
        None

    Return:
        paths (List): List of file paths
    '''
    title = 'Please choose a TRNSYS Input File'
    logging.info(title)
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
    '''This function presents a file dialog fora parametric table file.

    Args:
        None

    Return:
        path (str): File path
    '''
    title = 'Choose a parametric table, or cancel to perform '\
            'a regular simulation'
    logging.info(title)
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

    group1 = parser.add_argument_group('Important options', 'Use the ' +
                                       'following options to define how to ' +
                                       'run this program.')

    group1.add_argument('-d', '--deck', dest='dck', help='One or multiple ' +
                        'paths to TRNSYS input files. If not specified, a ' +
                        'file dialog opens instead', type=str, nargs='+')

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
                        'ROOT_FOLDER. This helps to prevent conflicts ' +
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
                        default='info')

    group2.add_argument('--sim_folder', action='store',
                        dest='sim_folder',
                        help='Folder where new simulations are created in, ' +
                        'if --copy_files is true or PARAMETRIC_TABLE is' +
                        ' given',
                        default=dck_proc.root_folder)

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

    # Read the user input:
    args = parser.parse_args()

    # Save user input by overwriting the default values:
    TRNExe.path_TRNExe = args.path_TRNExe
    TRNExe.mode_trnsys_hidden = args.mode_trnsys_hidden
    TRNExe.mode_exec_parallel = args.mode_exec_parallel
    TRNExe.n_cores = args.n_cores
    dck_proc.root_folder = os.path.abspath(args.sim_folder)
    dck_proc.regex_result_files = args.regex_result_files

    # Define the logging function
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=args.log_level.upper())

    if args.dck is None:
        dck_file_list = file_dialog_dck()
        if dck_file_list is None:
            logging.debug('Empty selection, exit program.')
            raise SystemExit
    else:
        # Get list of deck files (and convert relative into absolute paths)
        dck_file_list = [os.path.abspath(dck_file) for dck_file in args.dck]

    logging.debug('List of dck files:')
    if logging.getLogger().isEnabledFor(logging.DEBUG):
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
        dck_list = dck_proc.create_dcks_from_file_list(
                      dck_file_list,
                      update_dest=args.copy_files)

        if args.copy_files:  # Copy all required files to --sim_folder
            dck_proc.copy_assigned_files(dck_list)

    return dck_list  # Return a list of dck objects


if __name__ == "__main__":
    '''Main function
    This function is executed when the script is started directly with
    Python, not when it is loaded as a module.
    '''
    multiprocessing.freeze_support()  # Required on Windows

    trnexe = TRNExe()
    dck_proc = DCK_processor()
    dck_list = run_OptionParser(trnexe, dck_proc)
    dck_list = trnexe.run_TRNSYS_dck_list(dck_list)
    dck_proc.report_errors(dck_list)

    input('\nPress the enter key to exit.')
