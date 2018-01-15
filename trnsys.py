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
#from tqdm import tqdm
import psutil
import itertools
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
            raise FileNotFoundError('TRNExe.exe not found at ' + self.path_TRNExe)

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

        self.check_log_for_errors(dck)

        logging.debug('Finished PID ' + str(proc.pid))
        dck.success = True
        return dck

    def check_log_for_errors(self, dck):
        logfile_path = os.path.splitext(dck.file_path_dest)[0] + '.log'
        try:
            with open(logfile_path, 'r') as f:
                logfile = f.read()
        except FileNotFoundError:
            dck.error_msg_list.append('No logfile found')
            return

        re_msg = r'Fatal Error.*\n.*\n.*\n\s*(?P<msg>TRNSYS Message.*\n.*)'
        match_list = re.findall(re_msg, logfile)
#        print(match_list)
        for msg in match_list:
            dck.error_msg_list.append(msg)

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
        '''
        if len(dck_list) == 0:
            raise ValueError('The list of decks "dck_list" must not be empty.')

        start_time = time.time()
        if not self.mode_exec_parallel:
            return_list = []
            for dck in dck_list:
                ret = self.run_TRNSYS_dck(dck)
                return_list.append(ret)

        if self.mode_exec_parallel:
            if n_cores == 0:
                n_cores = min(multiprocessing.cpu_count() - 2, len(dck_list))

            logging.info('Parallel processing of ' + str(len(dck_list)) +
                         ' jobs on ' + str(n_cores) + ' cores')
            pool = multiprocessing.Pool(n_cores)

            '''For short lists, imap seemed the fastest option.
            With imap, the result is a comsumable iterator'''
            map_result = pool.imap(self.run_TRNSYS_dck, dck_list)
            '''With map_async, the results are available immediately'''
#            map_result = pool.map_async(self.run_TRNSYS_dck, dck_list)
#            return_list = pool.map(self.run_TRNSYS_dck, dck_list)

            pool.close()  # No more processes can be launched

            # Print progress counter
#            total = len(dck_list)/float(map_result._chunksize)
#            for result in tqdm(map_result, total=len(dck_list), unit=' Jobs'):
#                pass

#            total = len(dck_list)/float(map_result._chunksize)
#            total = len(dck_list)
#            print(map_result._chunksize)
#            with tqdm(total=total, unit=' Jobs') as pbar:
#                done_prev = 0
#                while map_result.ready() is False:
#                    done = total - map_result._number_left
#                    pbar.update(done-done_prev)
#                    done_prev = done
#                pbar.update(total - map_result._number_left - done_prev)

#            while map_result.ready() is False:
#                remaining = map_result._number_left
#                total = len(dck_list)/float(map_result._chunksize)
#                fraction = (total - remaining)/total
#                print('{:5.1f}% done'.format(fraction*100), end='\r')
#                time.sleep(1.0)
#                sys.stdout.flush()

            pool.join()  # Workers are removed

#            return_list = map_result.get()
#            return_list = map_result
            dck_list = map_result
#            print(return_list)

        script_time = pd.to_timedelta(time.time() - start_time, unit='s')
        logging.info('Finished all simulations in time: %s' % (script_time))

#        self.check_for_errors(dck_list)  # See if any simulations failed

#        return return_list
        return dck_list

#    def check_for_errors(self, dck_list):
#        for dck in dck_list:
#            if dck.success is False:
#                logging.error('Error in dck: ' + dck.file_name)
#                logging.error(dck.error_msg)


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
#        self.nested_dck_list = []
        self.replace_dict = dict()
        self.regex_dict = dict()
        self.regex_result_files = r'Result'
        self.assigned_files = []
        self.result_files = []

        self.load_dck_text()
        self.find_assigned_files()

    def load_dck_text(self):
        '''Here we store the complete text of the dck file as a property of
        the deck object. This may or may not prove to consume too much memory.
        '''
#        try:
        with open(self.file_path_orig, 'r') as f:
            self.dck_text = f.read()
#        except Exception as ex:
#            logging.error('File skipped! ' + str(ex))
#            break

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
#        self.regex_dict = dict()
#        self.parametric_table = pd.DataFrame()

    def read_parametric_table(self, param_table_file):
        parametric_table = self.read_filetypes(param_table_file)
#        print(param_df.)
#        combis = itertools.combinations(self.vals_active, r=2)
        parametric_table.set_index('hash', inplace=True)
        self.parametric_table = parametric_table
        return parametric_table

    def gen_row_hash(self):
        string = 'a'
        hash = hashlib.sha1(string).hexdigest()
        return hash

    def read_filetypes(self, filepath):
        '''Read any file type with stored data and return the Pandas DataFrame.
        Wrapper around Pandas' read_excel().
        '''
        filetype = os.path.splitext(os.path.basename(filepath))[1]
        if filetype in ['.xlsx', '.xls']:
            # Excel can be read automatically
            df = pd.read_excel(filepath)  # Pandas function
        elif filetype in ['.csv']:
            # Standard format: Here we guess everything. May or may not work
            df = pd.read_csv(filepath,
                             sep=None, engine='python',  # Guess separator
                             parse_dates=[0],  # Try to parse first column as date
                             infer_datetime_format=True)
        elif filetype in ['.out']:
            # Standard format: Here we guess everything. May or may not work
            df = pd.read_csv(filepath,
                             delim_whitespace=True,
                             parse_dates=[0],  # Try to parse first column as date
                             infer_datetime_format=True)
        else:
            raise NotImplementedError('Unsupported file extension: '+filetype)

        return df

    def get_parametric_dck_list(self, param_table, dck_file):
        dck_list = []
        for hash in self.parametric_table.index.values:
            dck = DCK(dck_file)
#            dirname = os.path.dirname(dck_file)
#            basename = os.path.basename(dck_file)
#            hash_file_path = os.path.join(dirname, hash, basename)
            dck.file_path_dest = os.path.join(self.path_root_folder,
                                              dck.file_name,
                                              hash,
                                              dck.file_name+'.dck')
            dck.hash = hash
#            nested_dck.file_path_orig = dck.file_path_orig
#            dck.nested_dck_list.append(nested_dck)
#            dck.hash_list.append(hash)
#            print(hash_file_path)

            for col in self.parametric_table.columns:
                dck.replace_dict[col] = self.parametric_table.loc[hash][col]
#            print(replace_dict_new)
            self.add_replacements_value_of_key(dck.replace_dict, dck)
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
            re_find = r'(?P<key>\b'+key+'\s=\s)(?P<value>\W*\d*\W?\d*\n)'
            # Replace match with key (capture group) plus the new value
            re_replace = r'\g<key>'+str(value)+'\n'
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

    def create_dcks_from_file_list(self, dck_file_list):
        dck_list = []
        for dck_file in dck_file_list:
            dck = DCK(dck_file)
            dck.file_path_dest = os.path.join(self.path_root_folder,
                                              dck.file_name,
                                              dck.file_name+'.dck')
            dck_list.append(dck)
        return dck_list

#    def rewrite_nested_dcks(self, dck_list):
#        dck_list_new = []
#        for dck in dck_list:
#            nested_dck_list_new = self.rewrite_dcks(dck.nested_dck_list)
#            print(nested_dck_list_new)
#            dck_list_new += nested_dck_list_new
#        return dck_list_new

    def rewrite_dcks(self, dck_list):
        '''Perform the replacements in self.replace_dict in the deck files.
        You have to use add_replacements(), add_replacements_value_of_key()
        or disable_plotters() before, to fill the replace_dict.

        Args:
            dck_list (list): List of paths to deck files to work on

        Returns:
            dck_list (list): List of paths
        '''
        # Process the dck file(s)
        for dck in dck_list:
            # Perform the replacements:
            for re_find, re_replace in dck.regex_dict.items():
                dck.dck_text = re.sub(re_find, re_replace, dck.dck_text)

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
        return

    def resample_using_Grouper(self, df, freq='2D'):
        level_values = df.index.get_level_values
        return (df.groupby([level_values(i) for i in [0, 1]]
                           + [pd.Grouper(freq=freq, level=-1)]).sum())


def run_OptionParser(TRNExe, dck_processor):
    '''Define and run the option parser. Set the user input and return the list
    of decks.
    '''
    parser = OptionParser()

    parser.add_option('-d', '--dck', action='store', type='string',
                      dest='dck_string',
                      help='TRNSYS deck file or list of deck files ' +
                      '(Example: "path\File1.dck path\File2.dck")')

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

#    logging.debug('args: ' + str(args))

    # Build the list of dck files. Can be stored in dck_string or in args:
    dck_list = []
    try:
        dck_list = options.dck_string.split()
    except Exception:
        if options.dck_string is not None:
            dck_list = [options.dck_string]
        pass

    for arg in args:
        if '.dck' in arg:
            dck_list.append(arg)

    logging.debug('List of dck files: '+str(dck_list))
    return dck_list


if __name__ == "__main__":
    '''Main function
    This function is executed when the script is started directly with
    Python, not when it is loaded as a module.
    '''
    multiprocessing.freeze_support()  # Required on Windows

    trnexe = TRNExe()
    dck_processor = DCK_processor()
    dck_list = run_OptionParser(trnexe, dck_processor)
    trnexe.run_TRNSYS_dck_list(dck_list)
