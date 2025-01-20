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

Module Core
-----------
This module defines the two classes ``DCK_processor()`` and ``TRNExe()``.
The first can create ``dck`` objects from regular TRNSYS
input (deck) files and manipulate them, the latter can run simulations with
the given ``dck`` objects.
"""

import logging
import math
import re
import os
import shutil
import multiprocessing
import threading
import subprocess
import time
from collections.abc import Sequence
import itertools
import datetime as dt
import psutil
import pandas as pd

# Default values that are used by multiple classes:
regex_result_files_def = r'Result|\.sum|\.pr.|\.plt|\.out'  # result files

# Define the logging function
logger = logging.getLogger(__name__)


class TRNExe():
    """Define the TRNExe class.

    The most prominent function a user will need is ``run_TRNSYS_dck_list()``,
    in order to perform the actual TRNSYS simulations with a list of ``dck``
    objects. All other functions are only used internally by that function.

    The behaviour of the TRNExe object (run in parallel, run hidden, number
    of CPU cores used) is controlled by the options given at initialization.
    """

    def __init__(self,
                 path_TRNExe=r'C:\Trnsys17\Exe\TRNExe.exe',
                 mode_trnsys_hidden=False,
                 mode_exec_parallel=False,
                 n_cores=0,
                 check_vital_sign=True,
                 pause_after_error=False,
                 delay=1,
                 check_log_after_sim=True,
                 cwd=None,
                 ):
        """Initialize the object.

        The optional argument n_cores allows control over the used CPU cores.
        By default, the total number of CPU cores minus one are used. This way
        your CPU is not under 100% load - and using all cores does not
        necessarily make total simulation time faster.

        Args:
            path_TRNExe (str, optional): Path to the actual TRNSYS executable

            mode_trnsys_hidden (bool): Run simulations completely hidden?

            mode_exec_parallel (bool): Run simulations in parallel?

            n_cores (int, optional): Number of CPU cores to use in parallel

            check_vital_sign (bool, optional): If ``False``, the check
            ``TRNExe_is_alive()`` is skipped. This is useful if you want to be
            able to pause the live plotter during a simulation.
            Default is ``True``.

            pause_after_error (bool, optional): Pause the following simulations
            if an error occured. Default is ``False``.

            delay (float, optional): Seconds of delay for each simulation
            start. Helps prevent I/O-errors that seem to occur when two
            TRNSYS processes are started at the same moment. Default is ``1``.

            cwd (str, optional): Path to be used as current working directory
            when running TRNEXE. Typically this could be set to the location
            of the .dck file that is simulated.

        Returns:
            None
        """
        self.path_TRNExe = path_TRNExe
        self.mode_trnsys_hidden = mode_trnsys_hidden
        self.mode_exec_parallel = mode_exec_parallel
        self.n_cores = n_cores
        self.check_vital_sign = check_vital_sign
        self.pause_after_error = pause_after_error
        self.delay = delay  # seconds delay for each simulation start
        self.check_log_after_sim = check_log_after_sim
        self.cwd = cwd

    def run_TRNSYS_dck(self, dck, delay=0):
        """Run a TRNSYS simulation with the given deck dck_file.

        Args:
            dck (dck object): A DCK object to simulate

            delay (float): A time in seconds

        Returns:
            dck (dck object): The simulated DCK object

        The ``delay`` in seconds is added before calling the TRNSYS
        executable. This should prevent ``I/O Error`` messages from TRNSYS
        that are otherwise sometimes occurring.
        """
        if dck.path_TRNExe is not None:
            path_TRNExe = dck.path_TRNExe
        else:
            path_TRNExe = self.path_TRNExe

        if not os.path.exists(path_TRNExe):
            raise FileNotFoundError('TRNExe.exe not found: '+path_TRNExe)

        if self.mode_trnsys_hidden:
            mode_trnsys = '/h'  # hidden mode
        else:
            mode_trnsys = '/n'  # batch mode

        time.sleep(delay)  # Start with a delay (seconds)

        proc = subprocess.Popen([path_TRNExe, dck.file_path_dest,
                                 mode_trnsys], cwd=self.cwd)

        logger.debug('TRNSYS started with PID %s (%s)', proc.pid,
                     dck.file_path_dest)

        # Wait until the process is finished
        try:
            while proc.poll() is None:
                # While we wait, we check if TRNSYS is still running
                if self.TRNExe_is_alive(proc.pid) is False:
                    logger.debug('TRNSYS is inactive and may have encountered '
                                 'an error, killing the process in 10 seconds '
                                 ': %s', dck.file_path_dest)
                    time.sleep(10)
                    proc.terminate()
                    dck.error_msg_list.append('Low CPU load - timeout')
                    dck.success = False
        except KeyboardInterrupt:
            print()
            logger.critical('Killing TRNSYS process %s', dck.file_path_dest)
            proc.terminate()
            proc.wait()

        # Check for error messages in the log and store them in the deck object
        if self.check_log_after_sim:
            if self.check_log_for_errors(dck) is False or dck.success is False:
                dck.success = False
                logger.debug('Finished PID %s with errors', proc.pid)

                if self.pause_after_error:  # and not self.mode_exec_parallel:
                    error_msgs = ''
                    for i, error_msg in enumerate(dck.error_msg_list):
                        error_msgs += '{}: {}\n'.format(i, error_msg)
                    logger.error('Errors in %s:\n%s', dck.file_path_dest,
                                 error_msgs)
                    input('Press enter to continue with next simulation.\n')

            else:
                logger.debug('Finished PID %s', proc.pid)
                dck.success = True
        # Return the deck object
        return dck

    def check_logs_for_errors(self, dck_list):
        """Check logs of all dcks in dck_list for errors."""
        for dck in dck_list:
            self.check_log_for_errors(dck)
        return dck_list

    def check_log_for_errors(self, dck):
        """Store errors in error_msg_list. Return False if errors are found.

        Function tries to open the log file of the given simulation and
        searches for error messages with a regular expression.

        Args:
            dck (dck object): Simulation to check for errors

        Returns:
            True/False (bool): Did simulation finish successfully?
        """
        logfile_path = os.path.splitext(dck.file_path_dest)[0] + '.log'
        try:
            with open(logfile_path, 'r') as f:
                logfile = f.read()
        except FileNotFoundError:
            dck.error_msg_list.append('No logfile found')
            return False

        # Capture the whole log as one DataFrame table
        re_msg = (r'(?P<severity>Notice|Warning|Fatal Error)'
                  r'\D*(?P<time>.*)\n'
                  r'    Generated by Unit     :\s+(?P<unit>.*)\n'
                  r'    Generated by Type     :\s+(?P<type>.*)\n'
                  r'.*Message \s+\d* : (?P<msg>.*\n.*)')

        match_list = re.findall(re_msg, logfile)
        df_log = pd.DataFrame(
            match_list,
            columns=["Severity", "Time", "Unit", "Type", "Message"],
            )
        df_log.index.set_names("No", inplace=True)
        df_log.set_index("Severity", append=True, inplace=True)
        df_log["Time"] = df_log["Time"].astype(float)
        # Clean up the message text, removing newlines and spaces
        df_log.replace("Not applicable or not available", float("nan"),
                       inplace=True)
        df_log.replace("\n", "", regex=True, inplace=True)
        df_log.replace(r"\s+", " ", inplace=True, regex=True)
        dck.df_log = df_log  # Store the log in the dck object

        # Capture warnings and errors in the old style with lists
        re_msg = r'Warning.*\n.*\n.*\n.*Message.*\:\ (?P<msg>.*\n.*)'
        match_list = re.findall(re_msg, logfile)
        if match_list:  # Matches of the regular expression were found
            for msg in match_list:
                dck.warn_msg_list.append(msg)  # Store warning messages

        re_msg = r'Fatal Error.*\n.*\n.*\n.*Message.*\:\ (?P<msg>.*\n.*)'
        match_list = re.findall(re_msg, logfile)
        if match_list:  # Matches of the regular expression were found
            for msg in match_list:
                dck.error_msg_list.append(msg)  # Store error messages
                dck.success = False
            return False
        else:  # No error messages found
            return True

    def TRNExe_is_alive(self, pid, interval_1=0.3, interval_2=9.7):
        """Check whether or not a particular TRNExe.exe process is alive.

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
        """
        if self.check_vital_sign is False:
            time.sleep(0.1)  # somehow this improves the simulation speed
            return True  # Skip the check if the user demands it

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

    def mp_init(self, lock):
        """Provide an initializer function for pool."""
        global starting
        starting = lock
        # global start_time
        # start_time = lock

    def run_TRNSYS_with_delay(self, dck, unused=0):
        """Run TRNSYS with a delay.

        Especially decks with TRNBuild seemed to cause problems, i.e. not
        start simulating, when they start at exactly the same time.

        This implements a solution discussed here
        https://stackoverflow.com/questions/30343018
        """
        # with start_time.get_lock():
        #     wait_time = max(0, start_time.value - time.time())
        #     time.sleep(wait_time)
        #     start_time.value = time.time() + self.delay
        #     # Now run the actual simulation
        #     dck = self.run_TRNSYS_dck(dck, delay=0)

        starting.acquire() # no other process can get it until it is released
        threading.Timer(self.delay, starting.release).start() # release
        dck = self.run_TRNSYS_dck(dck, delay=0)
        return dck

    def run_TRNSYS_dck_list(self, dck_list):
        """Run one TRNSYS simulation of each deck file in the dck_list.

        This is a wrapper around run_TRNSYS_dck() that allows execution in
        serial and in parallel.

        Args:
            dck_list (list): List of DCK objects to work on

        Returns:
            returned_dck_list (list): List of DCK objects worked on
        """
        if len(dck_list) == 0:
            raise ValueError('The list of decks "dck_list" must not be empty.')

        start_time = time.time()
        if not self.mode_exec_parallel:
            returned_dck_list = []
            for i, dck in enumerate(dck_list):
                print('{:5.1f}% done.'.format(i/len(dck_list)*100), end='\r')
                returned_dck = self.run_TRNSYS_dck(dck)
                returned_dck_list.append(returned_dck)
            print('100.0% done.', end='\r')  # Finish the last output with 100%

        if self.mode_exec_parallel:
            n_cores = self.n_cores
            if n_cores == 0:
                n_cores = max(
                    1,  min(multiprocessing.cpu_count() - 1, len(dck_list)))

            logger.info('Parallel processing of %s jobs on %s cores',
                        len(dck_list), n_cores)
            pool = multiprocessing.Pool(n_cores,
                                        initializer=self.mp_init,
                                        initargs=[multiprocessing.Lock()]
                                        # initargs=[multiprocessing.Value('d')]
                                        )

            # For short lists, imap seemed the fastest option.
            # With imap, the result is a consumable iterator
            # map_result = pool.imap(self.run_TRNSYS_dck, dck_list)
            # With starmap_async, the results are available immediately
            # delay_list = [x*self.delay for x in range(len(dck_list))]
            # After the first n_cores simulations, no delay is needed
            # delay_list[n_cores:] = [0] * len(delay_list[n_cores:])
            # map_result = pool.starmap_async(self.run_TRNSYS_dck,
            #                                 dck_list)
            # delay_list = [x*self.delay for x in range(len(dck_list))]
            delay_list = [0] * len(dck_list)
            # delay_list = [x*self.delay for x in range(len(dck_list))]
            # After the first n_cores simulations, no delay is needed
            # delay_list[n_cores:] = [0] * len(delay_list[n_cores:])
            # map_result = pool.starmap_async(self.run_TRNSYS_dck,
            #                                 dck_list)
            # map_result = pool.starmap_async(self.run_TRNSYS_with_delay,
                                            # zip(dck_list, delay_list))
            # map_result = pool.map_async(self.run_TRNSYS_dck,
            #                             dck_list)
            map_result = pool.map_async(self.run_TRNSYS_with_delay,
                                        dck_list)
            # map_result = pool.starmap_async(self.run_TRNSYS_dck,
            #                                 zip(dck_list, delay_list))

            pool.close()  # No more processes can be launched
            self.print_progress(map_result, start_time)
            pool.join()  # Workers are removed

            returned_dck_list = map_result.get()
            # With imap, the iterator must be turned into a list:
            # returned_dck_list = list(map_result)
            # print(returned_dck_list)

        script_time = pd.to_timedelta(time.time() - start_time, unit='s')
        script_time = str(script_time).split('.')[0]
        logger.info('Finished all simulations in time: %s', script_time)

        return returned_dck_list

    def print_progress(self, map_result, start_time):
        """Print info about the multiprocessing progress to the screen.

        Measure and print the percentage of simulations done and an
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
        """
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
            # Do the print. Use \r twice to make it show in Spyder
            # https://github.com/spyder-ide/spyder/issues/195
            print('\r'+text, end='\r')
            time.sleep(1.0)  # Sleep a certain number of seconds
        # Finish the last output with 100%
        t_elapsed = pd.to_timedelta(time.time() - start_time, unit='s')
        text = ('100.0% done. Time elapsed: {}                            '
                .format(str(t_elapsed).split('.')[0]))
        print('\r'+text)
        return


class DCK():
    """Define the deck class.

    Holds all the information about a deck file, including the content of the
    file itself. This allows manipulating the content, before saving the
    actual file to the old or a new location for simulation.
    These locations are stored, as well as the applied manipulations.
    After a simulation, potential errors can be stored in the object, too.
    """

    def __init__(self, file_path, regex_result_files=regex_result_files_def):
        self.file_path_orig = file_path
        self.file_path_dest = file_path
        self.file_name = os.path.splitext(os.path.basename(file_path))[0]
        self.error_msg_list = []  # List of all errors in the log file
        self.warn_msg_list = []  # List of all warnings in the log file
        self.df_log = pd.DataFrame(index=pd.MultiIndex.from_arrays(
            [[], []], names=('No', 'Severity')))  # Log file as a table
        self.success = None
        self.hash = None
        self.replace_dict = dict()
        self.regex_dict = dict()  # Regular expressions used for replacements
        self.regex_result_files = regex_result_files
        self.dck_text = ''
        self.dck_equations = dict()  # Dict with all equations in dck_text
        self.path_TRNExe = None

        # Perform the following functions to initialize some more values
        self.load_dck_text()
        self.find_assigned_files()

    def load_dck_text(self):
        """Store the complete deck file text as a property of the deck object.

        .. note::
            This may or may not prove to consume too much memory.

        Args:
            None

        Returns:
            None
        """
        if os.path.splitext(self.file_path_orig)[1] != '.dck':
            msg = self.file_path_orig+' has the wrong file type, must be .dck'
            raise ValueError(msg)
        else:
            # Deck files are not saved as UTF-8 by TRNSYS, so reading them
            # can cause problems. The following helps to prevent issues:
            with open(self.file_path_orig, 'r', encoding="WINDOWS-1252") as f:
                self.dck_text = f.read()

    def find_assigned_files(self):
        """Find all file paths that appear after an "ASSIGN" key in the deck.

        The files are sorted into input and output (=result) files by
        applying regex_result_files.
        The results are stored in the lists assigned_files and result_files.

        Args:
            None

        Returns:
            None
        """
        self.assigned_files = []
        self.result_files = []
        self.assigned_files = re.findall(r'ASSIGN \"(.*)\"', self.dck_text)
        for file in self.assigned_files.copy():
            # Test if the assigned file is supposed to be a result / output
            if re.search(self.regex_result_files, file):
                self.assigned_files.remove(file)
                self.result_files.append(file)

        if len(self.result_files) == 0:
            logger.warning('No result files were identified among the '
                           'assigned files in the deck %s. This may cause '
                           'issues. Is this regular expression correct? "%s"',
                           self.file_name, self.regex_result_files)

        # TRNSYS allows to name output files e.g. "***.plt", where the three
        # stars are a placeholder for the deck file name.
        for i, file in enumerate(self.result_files):
            file, n = re.subn(r'\*\*\*', repl=self.file_name, string=file)
            self.result_files[i] = file
            if n > 0:
                logger.debug('Replaced placeholder "***" with deck file '
                             'name in the assigned file %s', file)

    def find_equations(self, iteration=0, iter_max=10):
        """Find equations with key and value in the text of the deck file.

        Fill and return a dictionary with the results.
        This allows easy access to all properties of the simulation.
        The function tries to solve simple equations and turn the values
        into floats.

        Args:
            iteration (int): Number of the current iteration

            iter_max (int): The maximum number of iterations.

        Returns:
            dck_equations (dict): Key, value pairs of equations in dck_text
        """
        def gt(x, y):
            """Define custom TRNSYS function 'greater than'."""
            if x > y:
                return 1
            else:
                return 0

        def ge(x, y):
            """Define custom TRNSYS function 'greater or equal'."""
            if x >= y:
                return 1
            else:
                return 0

        def lt(x, y):
            """Define custom TRNSYS function 'lower than'."""
            if x < y:
                return 1
            else:
                return 0

        def le(x, y):
            """Define custom TRNSYS function 'lower or equal'."""
            if x <= y:
                return 1
            else:
                return 0

        def eql(x, y):
            """Define custom TRNSYS function 'equal'."""
            if x == y:
                return 1
            else:
                return 0

        def custom_not(x):
            """Define custom function to replace TRNSYS function 'not'."""
            if x == 1:
                return 0
            else:
                return 1

        def custom_and(x, y):
            """Define custom function to replace TRNSYS function 'and'."""
            if x and y:
                return 1
            else:
                return 0

        def custom_or(x, y):
            """Define custom function to replace TRNSYS function 'or'."""
            if x or y:
                return 1
            else:
                return 0

        def custom_sin(x):
            """Define custom function to replace TRNSYS function 'sin'."""
            return math.sin(math.radians(x))

        def custom_cos(x):
            """Define custom function to replace TRNSYS function 'sin'."""
            return math.cos(math.radians(x))

        def custom_tan(x):
            """Define custom function to replace TRNSYS function 'sin'."""
            return math.tan(math.radians(x))

        if iteration == 0:
            # Only find equations in the first iteration
            re_find = r'\n(?P<key>\b\S+)\s*=\s*(?P<value>.*?)(?=\s*\!|\s*\n)'

            match_list = re.findall(re_find, self.dck_text)
            if match_list:  # Matches of the regular expression were found
                for key, value in match_list:

                    # Some functions need a replacement to work with "eval()"
                    value = value.replace("not(", "custom_not(")
                    value = value.replace("and(", "custom_and(")
                    value = value.replace("or(", "custom_or(")
                    value = value.replace("^", "**")
                    value = value.replace("sin(", "custom_sin(")
                    value = value.replace("cos(", "custom_cos(")
                    value = value.replace("tan(", "custom_tan(")

                    try:  # Try to convert the string to a float
                        self.dck_equations[key] = float(value)
                    except Exception:
                        try:  # Try to solve an equation and turn it into float
                            self.dck_equations[key] = float(eval(value))
                        except Exception:  # Just store the string
                            self.dck_equations[key] = value

        # Work through all equations again. This gives a chance
        # to fill out variables and solve equations that failed before:
        for key, value in self.dck_equations.items():
            try:
                self.dck_equations[key] = float(eval(str(value)))
            except NameError as e:  # Equation contains a variable name
                name = re.findall(r"'(.+)'", str(e))[0]
                try:  # Try to find the value of the variable name
                    name_value = self.dck_equations[name]
                except KeyError:
                    pass
                else:  # Substitute the variable name with its value
                    eval_str = re.sub(name, str(name_value), value, count=1)
                    self.dck_equations[key] = eval_str  # Store result
                    try:  # Now try to convert it to a float again
                        self.dck_equations[key] = float(eval(eval_str))
                    except Exception:
                        pass
            except Exception:
                pass

        if iteration < iter_max:
            # Recursively solve more and more equations with each iteration
            iteration += 1
            self.find_equations(iteration=iteration, iter_max=iter_max)

        return self.dck_equations


class DCK_processor():
    """Define the deck processor class.

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
    """

    def __init__(self, sim_folder=r'C:\Trnsys17\Work\batch',
                 regex_result_files=regex_result_files_def):
        self.sim_folder = sim_folder
        self.regex_result_files = regex_result_files

    def parametric_table_auto(self, parametric_table, dck_file_list,
                              copy_files=True, disable_plotters=False):
        """Automate the creation of deck objects from a parametric table.

        Conveniently wrap the steps in the most common use case.
        A parametric table was given. Therefore we do the standard procedure
        of creating a ``dck`` object list from the parameters. We add those
        lists for all given files.

        Args:
            parametric_table (DataFrame): Pandas DataFrame

            dck_file_list (list or str): List of file paths, or single path
            string. If a list of file paths is used in combination with a
            parametric table, the ``hash`` property of each dck object
            is assigned a tuple containing the name of the dck and the
            index in the parametric table. This allows a unique
            identification.

            copy_files (bool, optional): Find and copy all assigned files
            from the source to the simulation folder. Default is ``True``.

            disable_plotters (bool, optional): If true, disable all plotters
            by setting their parameter 9 to '-1'. Even if TRNSYS is run in
            'hidden' mode, the plot data is still stored in files named e.g.
            D1_0.TMP during simulation, which can consume lots of storage
            space. Disabling the plotters prevents this. Default is ``False``.

        Returns:
            dck_list (list): List of dck objects
        """
        dck_list = []
        if (isinstance(dck_file_list, Sequence)
           and not isinstance(dck_file_list, str)):
            for dck_file in dck_file_list:
                dck_objs = self.get_parametric_dck_list(parametric_table,
                                                        dck_file)
                for dck in dck_objs:
                    # Make hash unique by including the name of deck file
                    dck.hash = (dck.file_name, dck.hash)
                dck_list += dck_objs
        else:  # Convert single dck_file path string to a list with one entry
            dck_list = self.get_parametric_dck_list(parametric_table,
                                                    dck_file_list)

        if disable_plotters:
            for dck in dck_list:
                self.disable_plotters(dck)

        self.rewrite_dcks(dck_list)
        if copy_files:
            self.copy_assigned_files(dck_list, find_files=True)
        return dck_list

    def parametric_table_read(self, param_table_file, **kwargs):
        """Read a parametric table from a given file.

        Return it as a DataFrame. Uses ``read_filetypes()`` to read the file.

        Args:
            param_table_file (str): Path to a file

        Returns:
            parametric_table (DataFrame): Pandas DataFrame
        """
        parametric_table = self.read_filetypes(param_table_file, **kwargs)

        if logger.isEnabledFor(logging.INFO):
            if not parametric_table.empty:
                logger.info('%s:', param_table_file)
                print(parametric_table)

        return parametric_table

    def parametric_table_combine(self, parameters):
        """Produce a parametric table from value combinations.

        Use all combinations of individual values of the given parameters.

        Args:
            parameters (dict): Dictionary with parameters and their values

        Returns:
            parametric_table (DataFrame): Pandas DataFrame
        """
        flat = [[(k, v) for v in vs] for k, vs in parameters.items()]
        combis = [dict(items) for items in itertools.product(*flat)]
        parametric_table = pd.DataFrame.from_dict(combis)

        if logger.isEnabledFor(logging.INFO):
            if not parametric_table.empty:
                logger.info('Parametric table from combinations:')
                print(parametric_table)

        return parametric_table

    def read_filetypes(self, filepath, **kwargs):
        """Read any file type with stored data and return the Pandas DataFrame.

        This is a wrapper around Pandas' read_excel() and read_csv().

        Please note: With 'kwargs', you can pass any (named) parameter down
        to the Pandas functions. The TRNSYS printer adds some useless rows
        at the top and bottom of the file? No problem, just define 'skiprows'
        and 'skipfooter'. For all options, see:
        http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
        """
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
                             sep=r'\s+',
                             encoding='WINDOWS-1252',  # TRNSYS encoding
                             **kwargs)
        elif filetype in ['.dat', '.txt']:
            logger.warning('Unsupported file extension: %s. Trying to read '
                           'it like a csv file.', filetype)
            df = pd.read_csv(filepath, **kwargs)
        else:
            raise NotImplementedError('Unsupported file extension: "' +
                                      filetype + '" in file ' + filepath)

        return df

    def get_parametric_dck_list(self, parametric_table, dck_file):
        """Create list of deck objects from ``parametric_table`` and dck_file.

        For each row in the parametric table, one deck object is created and
        parameters and their values in that row are prepared for replacement.

        After this function call, you will most likely also have to call:

        * ``dck_proc.rewrite_dcks(dck_list)``
        * ``dck_proc.copy_assigned_files(dck_list)``

        But you can manualy add other replacements before that, if required.
        ``parametric_table`` may also be a DataFrame with nothing but an
        index. This way you can define the unique indentifier ``hash`` used
        for each dck and perform more complex replacements manually.

        Args:
            parametric_table (DataFrame): Pandas DataFrame with parameters

            dck_file (str): Path to a TRNSYS deck file

        Returns:
            dck_list (list): List of ``dck`` objects
        """
        if isinstance(parametric_table, str):
            # Default is to hand over a parametric_table DataFrame. For
            # convenience, a file path is accepted and read into a DataFrame
            parametric_table = self.parametric_table_read(parametric_table)

        if parametric_table.index.duplicated().any():
            logger.error('Index of parameter table with duplicates:\n %s',
                         parametric_table.index)
            raise ValueError("The provided parameter table has duplicates "
                             "in its index. This is not allowed.")

        # Start building the list of deck objects
        dck_list = []
        for hash_ in parametric_table.index.values:
            # For each row in the table, one deck object is created and
            # updated with the correct information.
            dck = DCK(dck_file, regex_result_files=self.regex_result_files)
            dck.hash = hash_
            dck.file_path_dest = os.path.join(self.sim_folder,
                                              dck.file_name,
                                              str(dck.hash),
                                              dck.file_name+'.dck')

            # Store the replacements found in the deck object
            for col in parametric_table.columns:
                dck.replace_dict[col] = parametric_table.loc[hash_][col]

            # Some replacements are performed directly, others have
            # to be prepared (the type called "value of key")
            for key, value in dck.replace_dict.items():
                if 'ASSIGN' in key or 'LIMITS' in key:
                    # Use the replacement strings unchanged
                    self.add_replacements({key: value}, dck)
                else:
                    if key[0] == '!':  # First letter is '!'
                        continue  # Skip this key, it marks a comment

                    # Convert the replacements into regular expressions
                    self.add_replacements_value_of_key({key: value}, dck)

            # Done. Add the deck to the list.
            dck_list.append(dck)
        return dck_list

    def add_replacements(self, replace_dict_new, dck):
        """Add new entries to the existing replace_dict of a dck object.

        Basic use is to add strings, but also accepts regular expressions.
        These allow you to replace any kind of information.

        .. note::
            If a list of dck objects is given, the function is applied to
            all of them.

        Args:
            replace_dict_new (dict): Dictionary of 'str_old: string_new' pairs

            dck (DCK object or list): dck object(s) to add replacements to.

        Returns:
            None
        """
        if isinstance(dck, Sequence) and not isinstance(dck, str):
            for dck_obj in dck:  # Call this function recursively for each dck
                self.add_replacements(replace_dict_new, dck_obj)
        else:
            for re_find, re_replace in replace_dict_new.items():
                dck.regex_dict[re_find] = re_replace

    def add_replacements_value_of_key(self, replace_dict_new, dck):
        """Add new entries to the existing replace_dict of a dck object.

        Creates the required regular expression to replace the 'value' of the
        'key' in the deck file. Then calls add_replacements().

        .. note::
            If a list of dck objects is given, the function is applied to
            all of them.

        Args:
            replace_dict_new (dict): Dictionary of 'key: value' pairs

            dck (DCK object or list): dck object(s) to add replacements to.

        Returns:
            None

        Example:
            .. code::

                replace_dict_new = {'A_Koll': 550,
                                    'plot_on_off': -1}
        """
        for key, value in replace_dict_new.items():
            # Find key and previous value, possibly separated by '='
            re_find = r'(?P<key>\b'+key+r'\s?=\s?)(?P<value>.*)'
            # Replace match with key (capture group) plus the new value
            re_replace = r'\g<key>'+str(value)
            self.add_replacements({re_find: re_replace}, dck)

    def disable_plotters(self, dck):
        """Disable the plotters by setting their parameter 9 to '-1'.

        Calls add_replacements() with the required regular expressions.
        """
        re_find = r'\S+(?P<old_text>\s*! 9 Shut off Online )'
        re_replace = r''+str(-1)+r'\g<old_text>'
        self.add_replacements({re_find: re_replace}, dck)

    def reset_replacements(self, dck):
        """Reset all previous replacements.

        This is done by making ``replace_dict`` and ``regex_dict`` empty.
        """
        dck.replace_dict = dict()
        dck.regex_dict = dict()

    def create_dcks_from_file_list(self, dck_file_list, update_dest=False,
                                   copy_files=False):
        """Take a list of file paths and creates deck objects for each one.

        If the optional argument ``update_dest`` is True, the destination path
        becomes a folder with the name of the file, inside the ``sim_folder``
        (which is a property of the ``dck_processor`` object).
        The default False means simulating the deck in the original folder.

        .. note::
            Please make sure that the ``dck_file_list`` contains actual paths,
            not only file names.
            If the deck file is located in the same folder as the script,
            the path has to be ``"./TRNSYS_input_file.dck"``.

        .. note::
            With ``update_dest=True`` you will need to set ``copy_files=True``
            or call ``dck_proc.copy_assigned_files(dck_list)`` afterwards.

        Args:
            dck_list (list): List of paths to deck files to work on

            update_dest (bool, optional):

                * True: Simulate within ``sim_folder``
                * False: Simulate in the original folder (default)

            copy_files (bool, optional): If ``update_dest`` is True,
            this calls ``dck_proc.copy_assigned_files(dck_list)`` (which
            you would have to do manually otherwise!)

        Returns:
            dck_list (list): List of ``dck`` objects

        """
        dck_list = [DCK(dck_file, regex_result_files=self.regex_result_files)
                    for dck_file in dck_file_list]
        for dck in dck_list:
            dck.hash = dck.file_name  # use name as unique identifier

            if update_dest:
                dck.file_path_dest = os.path.join(self.sim_folder,
                                                  dck.file_name,
                                                  dck.file_name+'.dck')
        if update_dest and copy_files:
            self.copy_assigned_files(dck_list)

        return dck_list

    def rewrite_dcks(self, dck_list, print_warnings=True):
        """Perform the replacements in ``self.replace_dict`` in the deck files.

        You have to use add_replacements(), add_replacements_value_of_key()
        or disable_plotters() before, to fill the replace_dict.

        Args:
            dck_list (list): List of ``dck`` objects to work on

            print_warnings (bool, optional): Print warning for unsuccessful
            replacements. Default is True.

        Returns:
            dck_list (list): List of ``dck`` objects, now manipulated
        """
        # Process the deck file(s)
        for dck in dck_list:
            # Perform the replacements:
            for re_find, re_replace in dck.regex_dict.items():
                try:
                    dck.dck_text, number_of_subs_made = re.subn(re_find,
                                                                re_replace,
                                                                dck.dck_text)
                except Exception:
                    logger.error('Replace "%s" with "%s"', re_find, re_replace)
                    raise
                if number_of_subs_made == 0 and print_warnings:
                    logger.warning('Replacement not successful, '
                                   'because regex was not found: %s', re_find)

        return

    def copy_assigned_files(self, dck_list, find_files=False):
        """Copy all assigned files from the source to the simulation folder.

        All external files that a TRNSYS deck depends on are stored with
        the ASSIGN parameter. We make a list of those files and copy them
        to the required location. (Weather data, load profiles, etc.)

        Args:
            dck_list (list): List of ``dck`` objects to work on

            find_files (bool, optional): Perform ``find_assigned_files()``
            before doing the copy. Default is False.

        Returns:
            None
        """
        for dck in dck_list:
            if find_files:  # update 'dck.assigned_files'
                dck.find_assigned_files()  # search deck text for file paths
            # Copy the assigned files
            source_folder = os.path.dirname(dck.file_path_orig)
            destination_folder = os.path.dirname(dck.file_path_dest)
            for file in dck.assigned_files:
                source_file = os.path.join(source_folder, file)
                destination_file = os.path.join(destination_folder, file)
                # logger.debug('source      = '+source_file)
                # logger.debug('destination = '+destination_file)
                if not os.path.exists(os.path.dirname(destination_file)):
                    os.makedirs(os.path.dirname(destination_file))

                if not source_file == destination_file:
                    try:
                        shutil.copy2(source_file, destination_file)
                    except Exception as ex:
                        logger.error('%s: Error when trying to '
                                     'copy an "assigned" file (input data). '
                                     'The simulation may fail. Please check '
                                     'the argument --regex_result_files: '
                                     '"%s" '
                                     'Is this regular expression correct? It '
                                     'seperates output from input files.',
                                     dck.file_name, self.regex_result_files)
                        logger.error(ex)
                else:
                    logger.debug('%s: Copy source and destination are equal '
                                 'for file %s', dck.file_name, source_file)

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
        if logger.isEnabledFor(logging.DEBUG):
            list_dck_paths = [dck.file_path_dest for dck in dck_list]
            str_dck_paths = '\n'.join(list_dck_paths)
            logger.debug('List of copied dck files:\n%s', str_dck_paths)

        return

    def get_logs(self, dck_list):
        """Get complete logs stored in the ``dck`` objects of the dck_list.

        Args:
            dck_list (list): List of ``dck`` objects

        Returns:
            log_dict (dict): Dictionary of log messages for each dck
        """
        log_dict = dict(zip([dck.hash for dck in dck_list],
                            [dck.df_log for dck in dck_list]))
        return log_dict

    def report_warnings(self, dck_list, min_time=1.0):
        """Print all warnings stored in the ``dck`` objects of the dck_list.

        Args:
            dck_list (list): List of ``dck`` objects

            min_time (float): Only print warnings that occured after
            min_time. This provides focus on more important warnings.
            Default: 1.0 hours.

        Returns:
            log_dict (dict): Dictionary of log messages for each dck
        """
        log_list = []
        for dck in dck_list:
            try:
                logs = dck.df_log.xs("Warning", level="Severity",
                                     drop_level=False)
                logs = logs[logs["Time"] > min_time]
                if not logs.empty:
                    logger.warning('Warnings in %s:\n%s', dck.file_path_dest,
                                   logs.to_string())
            except KeyError:
                logs = dck.df_log[0:0]  # Use an empty dataframe
            log_list.append(logs)
        log_dict = dict(zip([dck.hash for dck in dck_list], log_list))
        return log_dict

    def report_errors_table(self, dck_list):
        """Print all errors stored in the ``dck`` objects of the dck_list.

        Reports the errors from the DataFrame object.

        Args:
            dck_list (list): List of ``dck`` objects

        Returns:
            log_dict (dict): Dictionary of log messages for each dck
        """
        log_list = []
        for dck in dck_list:
            try:
                logs = dck.df_log.xs("Fatal Error", level="Severity",
                                     drop_level=False)
                logger.warning('Errors in %s:\n%s', dck.file_path_dest,
                               logs.to_string())
            except KeyError:
                logs = dck.df_log[0:0]  # Use an empty dataframe
            log_list.append(logs)
        log_dict = dict(zip([dck.hash for dck in dck_list], log_list))
        return log_dict

    def report_errors(self, dck_list, warn=False):
        """Print all the errors stored in the ``dck`` objects of the dck_list.

        Args:
            dck_list (list): List of ``dck`` objects

            warn (bool): Optionally, raise a warning if any error was found

        Returns:
            error_found (bool): True, if an error was found
        """
        error_found = False
        for dck in dck_list:
            if dck.success is False:
                error_found = True
                error_msgs = ''
                for i, error_msg in enumerate(dck.error_msg_list):
                    error_msgs += '{}: {}\n'.format(i, error_msg)
                logger.error('Errors in %s:\n%s', dck.file_path_dest,
                             error_msgs)

        if warn and error_found:
            raise RuntimeWarning('Errors found in ' + dck.file_path_dest)

        return error_found

    def read_dck_results(self, dck, read_file_function):
        dck.result_data = dict()

        for result_file in dck.result_files:
            # Construct the full path to the result file
            result_file_path = os.path.join(os.path.dirname(
                                            dck.file_path_dest),
                                            result_file)
            # Use the provided function to read the file
            try:
                df = read_file_function(result_file_path)
            except Exception as ex:
                logger.error('Error when trying to read result file "%s"'
                             ': %s', result_file, ex)
                df = pd.DataFrame()

            dck.result_data[result_file] = df

        return dck

    def results_collect_parallel(self, dck_list, read_file_function,
        origin=None, n_cores=0, t_col='TIME'):
        """Collect the results of the simulations in parallel.

        This can be used instead of results_collect, but not all features
        are the same.
        """
        if n_cores == 0:
            n_cores = max(
                1,  min(multiprocessing.cpu_count() - 1, len(dck_list)))

        logger.info('Collecting %s results on %s cores',
                    len(dck_list), n_cores)
        pool = multiprocessing.Pool(n_cores)
        map_result = pool.starmap_async(self.read_dck_results,
                     zip(dck_list, [read_file_function]*len(dck_list)))
        pool.close()  # No more processes can be launched
        pool.join()  # Workers are removed
        dck_list = map_result.get()

        if isinstance(dck_list[0].hash, tuple):
            # If different deck files are simulated each with different
            # parameters, then the .hash value has a tuple of deck name
            # and hash number, which need to be assigned to columns
            hash_names = ['deck', 'hash']
        else:
            # Usually, the hash is a single value
            hash_names = ['hash']

        df_hashes = pd.DataFrame([dck.hash for dck in dck_list],
                                 columns=hash_names)

        param_table = pd.concat(
            [df_hashes,
             pd.DataFrame([dck.replace_dict for dck in dck_list])],
            axis='columns')

        logger.info('Concat DataFrames')
        result_data = dict()
        for result_file in dck_list[0].result_files:
            hash_list = [dck.hash for dck in dck_list]
            df_list = [dck.result_data[result_file] for dck in dck_list]
            df = pd.concat(df_list,
                           keys=[tuple(r) for r in param_table.to_numpy()],
                           names=list(param_table.columns))
            # Drop the last, unnamed, level
            df.index = df.index.droplevel(-1)

            if origin is not None:
                # Convert index to DateTime
                df[t_col] = pd.to_datetime(df[t_col], unit='h', origin=origin)

            df.set_index(t_col, append=True, inplace=True)
            result_data[result_file] = df

        return result_data



    def results_collect(self, dck_list, read_file_function, create_index=True,
                        origin=None, store_success=True, store_hours=True,
                        remove_leap_year=True, time_label_left=False,
                        keep_only_last_year=False):
        r"""Collect the results of the simulations.

        Combine the result files of the parametric runs into DataFrames. They
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

        A useful example code to process the results into one DataFrame.

        .. code:: python

            # Extract "group" names from the file names and create MultiIndex
            reg = r'Result\\(.*)\.(?:out|dat|sum)'
            files = [re.findall(reg, key)[0] for key in result_data.keys()]
            df = pd.concat(result_data.values(), axis='columns', keys=files)
            df.columns.set_names(['Group', 'Sensor'], inplace=True)

        Args:
            dck_list (list): A list of DCK objects

            read_file_function (func): A function that takes one argument (a
            file path) and returns a Pandas DataFrame.

            create_index (bool, optional): Move time and parameter columns to
            the index of the DataFrame. Default: True

            origin (str, optional): Start date for index (``'2003-01-01'``).
            Used if ``create_index=True``. Default is start of current year.

            store_success (bool, optional): Save boolean info about successful
            simulation in new column ``success``. Default: True

            store_hours (bool, optional): Save the original time column in
            a new column ``HOURS``. Default: True

            remove_leap_year (bool, optional): Try to remove February 29 from
            leap years. Default: True

            time_label_left (bool, optional): If true, label each time step
            on the left. By default, each time step is labelled on the right.
            For example, the first simulated hour may be labelled
            2022-01-01 00:00 (left) or 2022-01-01 01:00 (right).
            'right' is more in line with traditional labelling of weather data,
            while 'left' makes handling of the data with pandas easier.

            keep_only_last_year (bool, optional): If True, only the results
            from the last 8760 hours of the simulation results are kept.
            Default: False

        Returns:
            result_data (dict): A dictionary with one DataFrame for each file
        """
        result_data = dict()
        for i, dck in enumerate(dck_list, start=1):
            for result_file in dck.result_files:
                # Construct the full path to the result file
                result_file_path = os.path.join(os.path.dirname(
                                                dck.file_path_dest),
                                                result_file)
                # Use the provided function to read the file
                try:
                    df_new = read_file_function(result_file_path)

                    # Add the 'hash' and all the key, value pairs to DataFrame
                    if isinstance(dck.hash, tuple):
                        # If different deck files are simulated each with
                        # different parameters, then the .hash value has a
                        # tuple of deck name and hash number, which need to
                        # be assigned to columns
                        df_new[['deck', 'hash']] = dck.hash
                    else:
                        # Usually, the hash is a single value
                        df_new['hash'] = dck.hash

                    if store_success:
                        # Store simulation success
                        df_new['success'] = dck.success
                    for key, value in dck.replace_dict.items():
                        df_new[key] = value

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
                    logger.error('Error when trying to read result file "%s"'
                                 ': %s', result_file, ex)

            if logger.isEnabledFor(logging.INFO):  # Print progress
                frac = i/len(dck_list)*100
                print('\rCollecting results: {:5.1f}%'.format(frac), end='\r')

        logger.debug('Collected result files:')
        if logger.isEnabledFor(logging.DEBUG):
            for file in result_data.keys():
                print(file)

        if create_index:
            result_data = self.results_create_index(
                result_data, replace_dict=dck_list[0].replace_dict,
                origin=origin, store_hours=store_hours,
                remove_leap_year=remove_leap_year,
                time_label_left=time_label_left,
                keep_only_last_year=keep_only_last_year)

        for file, df in result_data.items():
            for column in df.columns:
                if column == 'success':
                    continue
                if df[column].dtype == 'object':
                    logger.error(
                        'Column "{}" in file "{}" has dtype "object". '
                        'This may be caused by extremely small numbers '
                        'like 1.234E-100 that TRNSYS writes as 1.234-100.'
                        .format(column, file))
                    # Try to fix this TRNSYS-bug by placing the missing "E":
                    df[column] = df[column].replace(r"(\d)(\-)(\d\d\d)",
                                                    r"\1E-\3", regex=True)
                    df[column] = pd.to_numeric(df[column])

        return result_data

    def results_create_index(self, result_data, replace_dict={}, origin=None,
                             time_names=['TIME', 'Time', 'Period'],
                             store_hours=True, remove_leap_year=True,
                             time_label_left=False, keep_only_last_year=False):
        """Put the time and parameter columns into the index of the DataFrame.

        The function expects the return of ``results_collect()``. This
        typically creates a multi-index and is arguably how DataFrames are
        supposed to be handled.

        .. note::
            TRNSYS simulations do not take leap years into account. Therefore
            we try to create an index without any February 29. If processed
            further, e.g. when resampling to weeks, this will be interpreted
            like a day with all entries equal to zero. Thus the week
            including 29.2. will have a lower energy sum. This is irritating,
            but better than the alternative, where 29.2. is filled with data
            but now the last day of the year is missing.

            The approach only works for simulations of one or multiple
            complete years. If this fails, a basic approach including 29.2.
            is used instead.

        Args:
            result_data (dict): The return of the function results_collect()

            replace_dict (dict): Is required to identify the parameter columns.
            Just use the replace_dict stored in one of your dck objects, e.g.

            .. code::

                replace_dict = dck_list[0].replace_dict

            origin (str, optional): Start date, e.g. ``'2003-01-01'``.
            TRNSYS does not care for years or days, but to get a pretty
            DataFrame we use datetime for the time column, which needs a
            fully defined date. Defaults to start of the current year.

            time_names (list, optional): List of column names that are
            used for identifying the time column.

            store_hours (bool, optional): Save the original time column in
            a new column ``HOURS``. Default: True

            remove_leap_year (bool, optional): Try to remove February 29 from
            leap years. Default: True

            time_label_left (bool, optional): If true, label each time step
            on the left. By default, each time step is labelled on the right.

            keep_only_last_year (bool, optional): If True, only the results
            from the last 8760 hours of the simulation results are kept.
            Default: False

        Returns:
            result_data (dict): A dictionary with one DataFrame for each file
        """
        if origin is None:  # Create default datetime object
            date = dt.datetime.today().replace(month=1, day=1)
            time_obj = dt.time(hour=0)
            origin = dt.datetime.combine(date, time_obj)
        else:
            origin = pd.Timestamp(origin)  # Convert string to datetime object

        for key, df in result_data.items():
            t_col = None
            for time_name in time_names:
                if time_name in df.columns:
                    t_col = time_name
            if t_col is None:
                continue  # Do not create index for this file
            # Convert TIME column to float
            df[t_col] = df[t_col].astype('float64')
            if store_hours:
                df['HOURS'] = df[t_col]  # Store the float hours

            if time_label_left and df[t_col][0] > 0:
                df[t_col] = df[t_col] - df[t_col][0]

            # Determine frequency as float, string and TimeDelta
            freq_float = df[t_col][1] - df[t_col][0]
            freq_str = str(freq_float)
            time_index = pd.to_datetime(df[t_col], unit='h', origin=origin)
            freq_timedelta = time_index[1] - time_index[0]

            if keep_only_last_year:
                keep_steps = int(8760 / freq_float)
                df.set_index(keys=['hash'], inplace=True)
                df_list = []
                for hash_ in df.index.get_level_values('hash').unique():
                    df_hash = df.xs(hash_, drop_level=False)
                    df_hash = df_hash[-keep_steps:]
                    df_hash.reset_index(inplace=True)
                    df_hash[t_col] = df_hash[t_col] - df_hash[t_col][0]
                    df_list.append(df_hash)
                df = pd.concat(df_list)

            # Convert the TIME float column to the datetime format
            if not remove_leap_year:
                df[t_col] = pd.to_datetime(df[t_col], unit='h', origin=origin)

            else:
                try:
                    # Approach where leap year days are not included in the
                    # resulting datetime column. This will fail if simulation
                    # did not run for one or multiple complete years.

                    # df is already the combined DataFrame of all simulations.
                    # Find the length of the longest simulation
                    n_hours = max(df[t_col])

                    # Check if data is suited for this approach
                    if n_hours % 8760 != 0 or len(df[t_col]) % n_hours != 0:
                        raise ValueError(
                            'Simulation data length is not one or multiple '
                            'complete years (8760 hours). Cannot remove leap '
                            'year days.')

                    n_years = int(n_hours/8760)  # Years per simulation
                    # Number of simulations (i.e. for different parameters)
                    n_sim = int(len(df[t_col])*freq_float/n_hours)

                    # Create date range, with the correct start and end date
                    end_date = origin.replace(year=origin.year + n_years)
                    if time_label_left is False:
                        dr = pd.date_range(start=origin + freq_timedelta,
                                           end=end_date, freq=freq_str+'h')
                    else:
                        dr = pd.date_range(start=origin, freq=freq_str+'h',
                                           end=end_date - freq_timedelta)
                    # Remove leap year days from the date range
                    dr = dr[~((dr.month == 2) & (dr.day == 29))]

                    # Copy the date range for each simulation
                    dr_copy = dr.copy()
                    for _ in range(n_sim - 1):
                        dr_copy = dr_copy.append(dr)
                    df[t_col] = dr_copy  # Insert date range into the DataFrame

                except Exception as ex:
                    logger.exception(ex)
                    logger.critical('Creating datetime index without removing '
                                    'leap year days.')

                    # Safe way of converting the float TIME column to datetime.
                    # However, this creates a 29.2. in all leap years
                    df[t_col] = pd.to_datetime(df[t_col], unit='h',
                                               origin=origin)

            # With a simulation time step of one minute, rounding errors
            # can produce an imperfect index. Rounding to seconds may fix it:
            df.set_index(keys=[t_col], inplace=True)  # convert to index
            df.index = df.index.round('s')  # Round to seconds
            df.reset_index(inplace=True)  # convert back to column

            # Create a list and use that as the new index columns
            idx_cols = [x for x in ['deck', 'hash'] if x in df.columns]
            idx_cols += list(replace_dict.keys()) + [t_col]
            df.set_index(keys=idx_cols, inplace=True)

            df.sort_index(inplace=True)

            # Check if the index has a frequency:
            df_test = df.copy()
            df_test = df_test.unstack(df_test.index.names[:-1])
            if pd.infer_freq(df_test.index) is None:
                logger.warning('Frequency of time index could not be inferred '
                               '(i.e. the time index is not spaced evenly)! '
                               'This can occur in simulations with one minute '
                               'time steps, where rounding errors add up. '
                               'Use "STEP = 0.016666666" (many decimal places'
                               ') for time steps of one minute in TRNSYS.')

            # Check if there are leap years in the data:
            bool_leap = df_test.index.get_level_values(t_col).is_leap_year
            if bool_leap[:-1].any():  # Ignore the very last time stamp
                df_leap = pd.DataFrame(data=bool_leap, index=df_test.index,
                                       columns=['is_leap_year'])
                df_leap['year'] = df_leap.index.get_level_values(t_col).year
                df_leap = df_leap[df_leap['is_leap_year']]
                years = set(df_leap['year'].astype(str))  # Set of unique years
                if remove_leap_year:
                    logger.warning(
                        key+': Data has leap years '+', '.join(years))

            result_data[key] = df  # df is not modified in place

        return result_data

    def results_slice_time(self, df, start, end):
        """Slice the time index from ``"start"`` to ``"end"``.

        Keep the other index columns intact. Expects the time column to be at
        last position in the multi-index. Date/time strings can be formatted
        e.g. ``"2018-01-01"`` or ``"2018-01-15 08:00"``.

        Args:
            df (DataFrame): Pandas DataFrame to slice

            start (str): Start date/time for slicing interval

            end (str): End date/time for slicing interval

        Returns:
            df_new (DataFrame): Sliced DataFrame
        """
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        n_index_cols = len(df.index.names)  # Number of index columns
        if n_index_cols == 1:
            df_new = df.loc[start:end]  # Slice regular index
        else:
            slice_list = [slice(None)]*(n_index_cols-1) + [slice(start, end)]
            slices = tuple(slice_list)  # Convert list to tuple
            df_new = df.loc[slices, :]  # Slice multiindex

        if len(df_new.index) == 0:
            logger.error('After slicing from %s to %s, data is empty! '
                         'Returning original data, which ranges from %s to %s',
                         start, end, df.index.min(), df.index.max())
            return df

        return df_new

    def results_resample(self, df, freq, regex_sum=r'^Q_|^E_',
                         regex_mean=r'^T_|^P_|^M_|^V_|COP',
                         prio='sum', level=None, **kwargs):
        """Resample a multi-indexed DataFrame to a new frequency.

        Expects the time column to be at last position in the multi-index.
        Columns not matched by the regular expressions will be dropped with
        an info message.

        .. note::
            Resampling for weeks with 'W' and months with 'M' will use actual
            calendar data (weeks do not necessarily begin on January 1.).

        Args:
            df (DataFrame): Pandas DataFrame to resample

            freq (str): Pandas frequency, e.g. ``"D"``, ``"W"``, ``"M"``
            or ``"Y"``. For a full list, see the Pandas documentation at
            http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

            regex_sum (str): Regular expression that matches all names of
            columns that should be summed up (e.g. energy values)

            regex_mean (str): Regular expression that matches all names of
            columns that should use the mean (e.g. temperatures)

            prio (str, optional): Set to ``"sum"`` (default) or ``"mean"`` to
            prioritise that regular expression. If a column fits to one regex,
            the other regex is not checked.

            level (str, optional): The name of a multiindex column level
            to use for comparison with the regular expressions

        kwargs:
            Additional keyword arguments, can be used to pass ``"closed"``
            and/or ``"label"`` (each with the values ``"left"`` or ``"right"``)
            to the ``"resample()"`` function

        Returns:
            df_new (DataFrame): Resampled DataFrame
        """
        # Apply the regular expressions to get two lists of columns
        cols_sum = []
        cols_mean = []
        cols_found = []
        for column in df.columns:
            if level is not None and isinstance(df.columns, pd.MultiIndex):
                column_str = column[df.columns.names.index(level)]
            else:
                column_str = column
            if prio == 'sum':
                if bool(re.search(regex_sum, column_str)):
                    cols_sum.append(column)
                    cols_found.append(column)
                elif bool(re.search(regex_mean, column_str)):
                    cols_mean.append(column)
                    cols_found.append(column)
                else:
                    logger.debug('Column %s does not match the regular '
                                 'expressions and is not resampled.', column)
            elif prio == 'mean':
                if bool(re.search(regex_mean, column_str)):
                    cols_mean.append(column)
                    cols_found.append(column)
                elif bool(re.search(regex_sum, column_str)):
                    cols_sum.append(column)
                    cols_found.append(column)
                else:
                    logger.debug('Column %s does not match the regular '
                                 'expressions and is not resampled.', column)
            else:
                raise ValueError('Resampling priority setting must be "sum" '
                                 'or "mean". "{}" is unknown.'.format(prio))

        if len(df.index.names) == 1:
            # If there is only the time column in the index, the resampling
            # is straight forward
            sum_df = df[cols_sum].resample(freq, **kwargs).sum(skipna=False)
            mean_df = df[cols_mean].resample(freq, **kwargs).mean()

        else:
            # Perform the resampling while preserving the multiindex,
            # achieved with the groupby function
            level_vls = df.index.get_level_values
            levels = range(len(df.index.names))[:-1]  # All columns except time

            if len(cols_sum) > 0:
                group = (df[cols_sum].groupby([level_vls(i) for i in levels]
                         + [pd.Grouper(freq=freq, **kwargs, level=-1)],
                         dropna=False))
                # sum(skipna=False) is required to make the sum of nan = nan
                # https://github.com/pandas-dev/pandas/issues/15675
                # sum_df = group.sum(skipna=False)  # TODO (not implemented)
                sum_df = group.mean() * group.count()  # workaround
            else:
                sum_df = pd.DataFrame()  # No sum() required, use empty df
            if len(cols_mean) > 0:
                mean_df = (df[cols_mean].groupby([level_vls(i) for i in levels]
                           + [pd.Grouper(freq=freq, **kwargs, level=-1)]
                           ).mean())
            else:
                mean_df = pd.DataFrame()  # No mean() required, use empty df

        # Recombine the two DataFrames into one
        df_new = pd.concat([sum_df, mean_df], axis=1)
        df_new = df_new[cols_found]  # Sort columns in their original order
        return df_new


class SimStudio():
    """Define the Simulation Studio class.

    Can be used to create a DCK file from a Trnsys Project File (.tpf).

    Uses the following command line options of Studio.exe:
      /d create deck file
      /r run simulation
      /q quit

    Other useful command line features of TRNSYS (not implemented):

    TRNBuild: Create VFM from command line
    subprocess.call(r'"C:\TRNSYS18\Building\TRNBuild.exe" "file.b18" /N /vfm')

    TRNBuild: Create SHM/ISM from command line
    subprocess.call(r'"C:\TRNSYS18\Building\TRNBuild.exe" "file.b18" /N /masks')
    """

    def __init__(self,
                 path_Studio=r'C:\Trnsys17\Studio\Exe\Studio.exe',
                 ):
        """Initialize a Simulation Studio object.

        Args:
            path_Studio (str, optional): Path to Simulation Studio executable

        Returns:
            None
        """
        self.path_Studio = path_Studio

    def create_dck_from_tpf(self, tpf, create=True, run=False, close=True,
                            silence_errors=False):
        """Create a dck file from the Trnsys Project File tpf.

        Args:
            tpf (str): Path to a Trnsys Project File.

            create (bool): Create a .dck file from the tpf. Default is True.

            run (bool): Run the simulation immediately. Be careful, a
            simulation will be started, but Python code will not wait for the
            simulation to finish. Use the TRNExe class for reliably running
            simulations. Default is False.

            close (bool): Close Simulation Studio automatically. Defaul is True

        Returns:
            ret (int): A return value of 0 indicates success

        """
        args = [self.path_Studio, os.path.abspath(tpf)]
        if create:
            args.append('/d')
        if run:
            args.append('/r')
            logger.warning("A simulation will be started, but Python code "
                           "not wait for the simulation to finish.")
        if close:
            args.append('/q')

        logger.info("Calling Simulation Studio with command '%s'",
                    ' '.join(args))
        ret = subprocess.call(args)
        if ret != 0 and not silence_errors:
            raise ValueError("Simulation Studio returned error")
        return ret
