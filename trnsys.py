# -*- coding: utf-8 -*-
"""
Created on Mo Jan  8 08:07:52 2018

@author: Joris Nettelstroth

Two use cases are possible:
    1) Run this script from the command line. The main function is called,
       which uses the option parser for the user input.
       Type 'python trnsys.py --help' to see the help message and the options.
    2) Import this script as a module into your own Python script. There you
       can initialize an object of the TRNExe() class and use its functions
"""

from optparse import OptionParser  # For parsing options with the program call
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


class TRNExe(object):
    '''The TRNExe class.
    '''

    def __init__(self,
                 path_TRNExe=r'C:\Trnsys17\Exe\TRNExe.exe',
                 mode_trnsys_hidden=False,
                 mode_exec_parallel=False
                 ):
        self.path_TRNExe = path_TRNExe
        self.mode_trnsys_hidden = mode_trnsys_hidden
        self.mode_exec_parallel = mode_exec_parallel

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
                logging.debug('\nTRNSYS may have encountered an error, ' +
                              'killing the process in 10 seconds: ' +
                              dck.file_path_dest)
                time.sleep(10)
                proc.terminate()
                dck.error_msg_list.append('Low CPU load - timeout')
                dck.success = False
                return dck

        logging.debug('Finished PID ' + str(proc.pid))

        # Check for error messages in the log and store them in the deck object
        if self.check_log_for_errors(dck) is False:
            dck.success = False
        else:
            dck.success = True
        # Return the deck object
        return dck

    def check_log_for_errors(self, dck):
        '''Stores errors in error_msg_list. Returns False if errors are found.
        '''
        logfile_path = os.path.splitext(dck.file_path_dest)[0] + '.log'
        try:
            with open(logfile_path, 'r') as f:
                logfile = f.read()
        except FileNotFoundError:
            dck.error_msg_list.append('No logfile found')
            return False

        re_msg = r'Fatal Error.*\n.*\n.*\n\s*(?P<msg>TRNSYS Message.*\n.*)'
        match_list = re.findall(re_msg, logfile)
        if match_list:
            for msg in match_list:
                dck.error_msg_list.append(msg)
            return False
        else:
            return True

    def TRNExe_is_alive(self, pid, interval=1.0):
        '''Check whether or not a particular TRNExe.exe process is alive.
        This status is guessed by measuring its CPU load over the given time
        interval.
        '''
        try:
            # The process might have finished by the time we get to check
            # its status. This would cause an error.
            p = psutil.Process(pid)
            if p.cpu_percent(interval=interval) < 1.0:
                # print(p.memory_info())
                return False
        except Exception:
            pass

        return True

    def run_TRNSYS_dck_list(self, dck_list, n_cores=0):
        '''Run one TRNSYS simulation of each deck file in the dck_list.
        This is a wrapper around run_TRNSYS_dck() that allows execution in
        serial and in parallel.
        The optional argument n_cores allows control over the used CPU cores.
        By default, the total number of CPU cores minus one are used. This way
        your CPU is not under 100% load - and using all cores does not
        necessarily make total simulation time faster.

        Args:
            dck_list (list): List of DCK objects to work on

            n_cores (int, optional): Number of CPU cores to use

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
        return


class DCK(object):
    '''Deck class.
    '''
    def __init__(self, file_path):
        self.file_path_orig = file_path
        self.file_path_dest = file_path
        self.file_name = os.path.splitext(os.path.basename(file_path))[0]
        self.error_msg_list = []
        self.success = None
        self.hash = None
        self.replace_dict = dict()
        self.regex_dict = dict()
        self.regex_result_files = r'Result'
        self.assigned_files = []
        self.result_files = []
        self.dck_text = ''

        # Perform the following functions to initialize some more values
        self.load_dck_text()
        self.find_assigned_files()

    def load_dck_text(self):
        '''Here we store the complete text of the deck file as a property of
        the deck object.
        HINT: This may or may not prove to consume too much memory.
        TODO: Decide whether exceptions should be caught or raised + Should
        I check for e.g. the right extension? (.dck, not .tpf)
        '''
        try:
            with open(self.file_path_orig, 'r') as f:
                self.dck_text = f.read()
        except Exception as ex:
            logging.error('File skipped! ' + str(ex))

    def find_assigned_files(self):
        self.assigned_files = re.findall(r'ASSIGN \"(.*)\"', self.dck_text)
        for file in self.assigned_files.copy():
            if re.search(self.regex_result_files, file):
                self.assigned_files.remove(file)
                self.result_files.append(file)

    def get_errors(self):
        return


class DCK_processor(object):
    '''Deck processor class.
    '''
    def __init__(self, path_root_folder=r'C:\Trnsys17\Work\batch'):
        self.path_root_folder = path_root_folder

    def auto_parametric_table(self, parametric_table, dck_file_list):
        # A parametric table was given. Therefore we do the standard procedure
        # of creating a deck list from the parameters. We add those lists for
        # all given files
        dck_list = []
        for dck_file in dck_file_list:
            dck_list += dck_proc.get_parametric_dck_list(parametric_table,
                                                         dck_file)
        dck_proc.rewrite_dcks(dck_list)
        dck_proc.copy_assigned_files(dck_list)
        return dck_list

    def read_parametric_table(self, param_table_file):
        parametric_table = self.read_filetypes(param_table_file)
#        print(param_df.)
#        combis = itertools.combinations(self.vals_active, r=2)
#        parametric_table.set_index('hash', inplace=True)

        logging.info(param_table_file+':')
        if logging.getLogger().isEnabledFor(logging.INFO):
            print(parametric_table)

        return parametric_table

    def gen_row_hash(self):
        '''Unused'''
        string = 'a'
        hash = hashlib.sha1(string).hexdigest()
        return hash

    def read_filetypes(self, filepath, **args):
        '''Read any file type with stored data and return the Pandas DataFrame.
        Wrapper around Pandas' read_excel() and read_csv().
        Please note: With 'args', you can pass any (named) parameter down
        to the Pandas functions. The TRNSYS printer adds some useless rows
        at the top and bottom of the file? No problem, just define 'skiprows'
        and 'skipfooter'. For all options, see:
        http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
        '''
        filetype = os.path.splitext(os.path.basename(filepath))[1]
        if filetype in ['.xlsx', '.xls']:
            # Excel can be read automatically
            df = pd.read_excel(filepath, **args)  # Pandas function
        elif filetype in ['.csv']:
            # Standard format: Here we guess everything. May or may not work
            df = pd.read_csv(filepath,
                             sep=None, engine='python',  # Guess separator
                             parse_dates=[0],  # Try to parse column 0 as date
                             infer_datetime_format=True,
                             **args)
        elif filetype in ['.out']:
            # Standard format for TRNSYS: Separator is whitespace
            df = pd.read_csv(filepath,
                             delim_whitespace=True,
                             **args)
        elif filetype in ['.dat', '.txt']:
            logging.warning('Unsupported file extension: ' + filetype +
                            '. Trying to read it like a csv file.')
            df = pd.read_csv(filepath, **args)
        else:
            raise NotImplementedError('Unsupported file extension: '+filetype)

        return df

    def get_parametric_dck_list(self, parametric_table, dck_file):
        if isinstance(parametric_table, str):
            # Default is to hand over a parametric_table DataFrame. For
            # convenience, a file path is accepted and read into a DataFrame
            parametric_table = self.read_parametric_table(parametric_table)

        # Start building the list of deck objects
        dck_list = []
        for hash in parametric_table.index.values:
            # For each row in the table, one deck object is created and
            # updated with the correct information.
            dck = DCK(dck_file)
            dck.hash = str(hash)
            dck.file_path_dest = os.path.join(self.path_root_folder,
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
            re_find = r'(?P<key>\b'+key+'\s=\s)(?P<value>.*)'
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
        is updated and based on the 'path_root_folder'.
        The default False means simulating the deck in the original folder.
        '''
        dck_list = [DCK(dck_file) for dck_file in dck_file_list]
        if update_dest:
            for dck in dck_list:
                dck.file_path_dest = os.path.join(self.path_root_folder,
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

                shutil.copy2(source_file, destination_file)

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
        logging.debug('List of parametric dck files:')
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            for dck in dck_list:
                print(dck.file_path_dest)

        return

    def report_errors(self, dck_list):
        for dck in dck_list:
            if dck.success is False:
                print('Errors in ' + dck.file_path_dest)
                for i, error_msg in enumerate(dck.error_msg_list):
                    print('  '+str(i)+': '+error_msg)
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

        def read_file_function(result_file_path):
            return dck_proc.read_filetypes(result_file_path)

        This means that you can utilize the existing read_filetypes(), which
        can already handle different file types. Please see the docs for
        read_filetypes() for info about passing over additional arguments to
        customize it to your needs.

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
                df_new = read_file_function(result_file_path)

                # Add the 'hash' and all the key, value pairs to the DataFrame
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
            df = df.sort_index()

        return result_data

    def results_resample(self, df, freq):
        '''Resample a multi-indexed DataFrame to a new frequency.
        Expects the time column to be at last position in the multi-index.
        '''
        level_values = df.index.get_level_values
        levels = range(len(df.index.names))[:-1]  # All columns except time
        return (df.groupby([level_values(i) for i in levels]
                           + [pd.Grouper(freq=freq, level=-1)]).sum())

    def mark_index_for_DataExplorer(self, df):
        '''Put '!' in front of index column names, to mark them as
        classifications. Is not applied to time columns.
        '''
        idx_cols_rename = []
        for name in df.index.names:
            if name not in ['TIME', 'Time', 'time']:
                idx_cols_rename.append('!'+name)
            else:
                idx_cols_rename.append(name)

        print(idx_cols_rename)
        df.index = df.index.rename(idx_cols_rename)
        return df


def run_OptionParser(TRNExe, dck_processor):
    '''Define and run the option parser. Set the user input and return the list
    of decks.
    '''
    parser = OptionParser()

    parser.add_option('-d', '--dck', action='store', type='string',
                      dest='dck_string',
                      help='TRNSYS deck file or list of deck files ' +
                      '(Example: "path\File1.dck path\File2.dck")')

    parser.add_option('-t', '--table', action='store', type='string',
                      dest='parametric_table',
                      help='Path to a parametric table file with ' +
                      'replacements to be made in the given deck files',
                      default=None)

    parser.add_option('-i', '--hidden', action='store_true',
                      dest='mode_trnsys_hidden',
                      help='Hide all TRNSYS windows; default = %default',
                      default=TRNExe.mode_trnsys_hidden)

    parser.add_option('-p', '--parallel', action='store_true',
                      dest='mode_exec_parallel',
                      help='Run simulations in parallel; default = %default',
                      default=TRNExe.mode_exec_parallel)

    parser.add_option('-l', '--log_level', action='store', dest='log_level',
                      help='log-level can be one of: debug, info, ' +
                      'warning, error or critical; default = %default',
                      default='warning')

    parser.add_option('-r', '--root_folder', action='store',
                      dest='path_root_folder',
                      help='Root folder where all new simulations are ' +
                      'created in; default = %default',
                      default=dck_processor.path_root_folder)

    # Read the user input:
    options, args = parser.parse_args()

    # Save user input by overwriting the default values:
    TRNExe.mode_trnsys_hidden = options.mode_trnsys_hidden
    TRNExe.mode_exec_parallel = options.mode_exec_parallel
    dck_processor.path_root_folder = options.path_root_folder

    # Define the logging function
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=options.log_level.upper())

    # Build the list of dck files. Can be given in dck_string or in args:
    dck_file_list = []
    try:
        dck_file_list = options.dck_string.split()
    except Exception:
        if options.dck_string is not None:
            dck_file_list = [options.dck_string]
        pass

    for arg in args:
        if '.dck' in arg:
            dck_file_list.append(arg)

    logging.debug('List of dck files:')
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        for dck_file in dck_file_list:
            print(dck_file)

    # All the user input is read in. Now the appropriate action is taken
    if options.parametric_table is not None:
        # A parametric table was given. Automate the default procedure
        dck_list = dck_proc.auto_parametric_table(options.parametric_table,
                                                  dck_file_list)
    else:
        # Just a list of decks to simulate, without any modifications
        dck_list = dck_proc.create_dcks_from_file_list(dck_file_list)

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
