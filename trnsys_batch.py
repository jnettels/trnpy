# -*- coding: utf-8 -*-
'''
TRNpy: Parallelized TRNSYS simulation with Python
=================================================

**Examples for Python integration**

By importing trnpy.py as a module into your own Python script, you get full
control over its functions and can integrate it into your workflow. This
approach also allows you to (to some degree) automatically collect, combine
and evaluate the simulation results.

You can initialize objects of the ``DCK_processor()`` and ``TRNExe()`` classes
and use their functions. The first can create ``dck`` objects from regular
TRNSYS input (deck) files and manipulate them, the latter can run simulations
with the given ``dck`` objects.

Post-processing of the results is possible, but will always require adaptions
for your specific situation.

'''

import logging
import trnpy
import multiprocessing
import pandas as pd

'''This allows to import a module from any location'''
#sys.path.append(os.path.abspath('somewhere'))
#import something  # A module located at 'somewhere'


def trnsys_batch_example_01(dck_file):
    '''The first example for batch execution of TRNSYS with the TRNpy module.
    We have an Excel file with combinations of parameter values and want one
    TRNSYS simulation for each row in the table.
    '''
    # Create a DCK_processor object. It gives us methods to create and
    # manipulate the deck files that we want to work with
    dck_proc = trnpy.DCK_processor()

    # Load a parametric table located in the current directory
    param_table_file = r'Parametrics.xlsx'
    param_table = dck_proc.parametric_table_read(param_table_file)

    # Modify the table on demand (select only certain rows)
#    param_table = param_table.loc[0:1]
#    print(param_table)

    # Create a deck list from the parameters and the original deck file
    dck_list = dck_proc.get_parametric_dck_list(param_table, dck_file)

    # Disable the plotters
    for dck in dck_list:
        dck_proc.disable_plotters(dck)

    # Apply parametric modifications to list of decks
    dck_proc.rewrite_dcks(dck_list)

    # Copy all files
    dck_proc.copy_assigned_files(dck_list)

    # Create a TRNSYS object
    trnexe = trnpy.TRNExe(
                          mode_exec_parallel=True,
#                          mode_trnsys_hidden=True,
#                          n_cores=7,
                          )

    # Run the TRNSYS simulations
    dck_list = trnexe.run_TRNSYS_dck_list(dck_list)

    # Report any errors that occured
    dck_proc.report_errors(dck_list)

    # Post-processing: This is where the example gets quite specific
    def read_file_function(result_file_path):
        '''Different TRNSYS printer outputs are formatted differently. But
        we can still use the existing read_filetypes(), we only have to provide
        some additional arguments for certain result files.
        '''
        if 'simsum' in result_file_path:
            # Adapt read_filetypes for these files
            return dck_proc.read_filetypes(result_file_path,
                                           skiprows=33,
                                           skipfooter=2,
                                           engine='python')
        elif 'Speicher.out' in result_file_path:
            raise ValueError("Skip Speicher.out (not needed)")
        else:  # All other files can be read automatically
            return dck_proc.read_filetypes(result_file_path)

    # Collect the results of the simulations. Our goal is to combine
    # the result files of the parametric runs into DataFrames.
    result_data = dck_proc.results_collect(dck_list, read_file_function)

    # Put the time and parameter columns into the index of the DataFrame.
#    data_start_date = '2003-01-01'
    data_start_date = '2005-01-01'
    result_data = dck_proc.results_create_index(result_data,
                                                dck_list[0].replace_dict,
                                                data_start_date)

    # Now we have got all the results in memory. We can call them by their file
    # paths (as assigned in TRNSYS).
    df_hour = result_data[r'Result\temperaturen_1.out']
#    print(df_hour)

    # Slicing the index for a new time interval
#    df_hour = dck_proc.results_slice_time(df_hour, '2005-01-01', '2006-01-01')
    df_hour = dck_proc.results_slice_time(df_hour, '2006-01-01', '2008-01-01')
#    print(df_hour)

    # Now depending on our needs, we can also group the data by week or year
#    freq = 'D'
#    freq = 'W'
    freq = 'M'
#    freq = 'Y'
    df_resample = dck_proc.results_resample(df_hour, freq=freq,
                                            regex_mean=r'T_|M_|WP_COP')
    print(df_resample)

    df_energy_1 = result_data[r'Result\simsum_energie_1.out']
    df_energy_2 = result_data[r'Result\simsum_energie_2.out']
    df_energy = pd.concat([df_energy_1, df_energy_2], axis=1)
    df_energy.drop(columns=['Month', 'hash', 'hash.1'], inplace=True)
#    print(df_energy)

    # With the manipulation completed, we have the option to view the result:
    dck_proc.DataExplorer_open(df_resample)
#    dck_proc.DataExplorer_open(df_energy)


def trnsys_batch_example_02(dck_file_list):
    # Create a DCK_processor object. It gives us methods to create and
    # manipulate the deck files that we want to work with
    dck_proc = trnpy.DCK_processor()

    # Convert the list of file paths to decks into a list of deck objects.
    # These objects will later contain information on the success and
    # results of the simulation
    dck_list = dck_proc.create_dcks_from_file_list(dck_file_list,
                                                   update_dest=True)

    for dck in dck_list:
        # Now we define a couple of replacements that should take place in
        # the deck files.
        # 1) Most replacements in the deck file are of the type 'key = value'
        replace_dict = {'on_off': -1,
                        'WP_on_off': 1,
                        }
        dck_proc.add_replacements_value_of_key(replace_dict, dck)

        # 2) In some places of the file, we just want to replace some strings
        replace_dict = {r'([^P]).out\"': r'\1_WP.out"',
                        }
        dck_proc.add_replacements(replace_dict, dck)

        # 3) There is a convenience function to disable all plotters
        dck_proc.disable_plotters(dck)

    # Internally, each dck has stored all those modifications.
    # Now we rewrite the content of the deck files
    dck_proc.rewrite_dcks(dck_list)

    # We copy the decks to their new destination:
    # We must copy all the 'assigned' files as defined in the decks,
    # e.g. weather data, load profiles, etc. to the simulation folder
    dck_proc.copy_assigned_files(dck_list)

    # Create a TRNSYS object
    trnexe = trnpy.TRNExe(
                          mode_exec_parallel=True,
                          mode_trnsys_hidden=True,
                          n_cores=7,
                          )

    # Run the TRNSYS simulations
    dck_list = trnexe.run_TRNSYS_dck_list(dck_list)


if __name__ == "__main__":
    '''Main function
    This function is executed when the script is started directly with
    Python, not when it is loaded as a module.
    '''
    multiprocessing.freeze_support()  # Required on Windows

    # Global Pandas option for displaying terminal output
    pd.set_option('display.max_columns', 0)

    # Define the logging function
    logging.basicConfig(format='%(asctime)-15s %(message)s', level='DEBUG')
#    logging.basicConfig(format='%(asctime)-15s %(message)s', level='INFO')
#    logging.basicConfig(format='%(asctime)-15s %(message)s', level='ERROR')

    dck_file_list = [
        #    r'Steinfurt_180105\Steinfurt_180105.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test1.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test2.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test3.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test4.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test5.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test6.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test7.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test8.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test9.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test10.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test11.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test12.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test13.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test14.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test15.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test16.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test17.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test18.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test19.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_171204.dck',
        #    r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_171201.dck',
        #    r'C:\Trnsys17\Work\futureSuN\HK\Hannover_Koll_Multi_170222.dck',
        r'C:\Trnsys17\Work\futureSuN\HK\Hannover_Koll_Multi_180220.dck',
        ]

    # This example takes only a single deck file path as input
    for dck_file in dck_file_list:
        trnsys_batch_example_01(dck_file)

    # This example takes a list of deck file paths as input
#    trnsys_batch_example_02(dck_file_list)
