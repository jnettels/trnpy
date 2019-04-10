# -*- coding: utf-8 -*-
'''
**TRNpy: Parallelized TRNSYS simulation with Python**

Copyright (C) 2019 Joris Nettelstroth

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.


TRNpy: Parallelized TRNSYS simulation with Python
=================================================
Simulate TRNSYS deck files in serial or parallel and use parametric tables to
perform simulations for different sets of parameters. TRNpy helps to automate
these and similar operations by providing functions to manipulate deck files
and run TRNSYS simulations from a programmatic level.

Examples batch processing
-------------------------

By importing trnpy as a module into your own Python script, you get full
control over its functions and can integrate it into your workflow. This
approach also allows you to (to some degree) automatically collect, combine
and evaluate the simulation results.

You can initialize objects of the ``DCK_processor()`` and ``TRNExe()`` classes
and use their functions. The first can create ``dck`` objects from regular
TRNSYS input (deck) files and manipulate them, the latter can run simulations
with the given ``dck`` objects.

Post-processing of the results is possible, but will require adaptions
for your specific situation.

'''

import logging
import multiprocessing
import pandas as pd
import trnpy.core  # TRNpy core module


def main():
    '''Main function
    '''
    multiprocessing.freeze_support()  # Required on Windows

    # Global Pandas option for displaying terminal output
    pd.set_option('display.max_columns', 0)

    # Define the logging function
    logging.basicConfig(format='%(asctime)-15s %(message)s', level='DEBUG')
#    logging.basicConfig(format='%(asctime)-15s %(message)s', level='INFO')
#    logging.basicConfig(format='%(asctime)-15s %(message)s', level='ERROR')

    # Define a list of deck files to work on. We choose one of the official
    # examples. Note: You have to use Simulation Studio to create the .dck
    # from the .tpf file!
    dck_file_list = [
        r'C:\Trnsys17\Examples\Photovoltaics\PV-Inverter.dck',
        ]

    # This example takes only a single deck file path as input
    for dck_file in dck_file_list:
        trnsys_batch_example_01(dck_file)

    # This example takes a list of deck file paths as input
#    trnsys_batch_example_02(dck_file_list)


def trnsys_batch_example_01(dck_file):
    '''The first example for batch execution of TRNSYS with the TRNpy module.
    We have an Excel file with combinations of parameter values and want one
    TRNSYS simulation for each row in the table.
    '''
    # Create a DCK_processor object. It gives us methods to create and
    # manipulate the deck files that we want to work with
    dck_proc = trnpy.core.DCK_processor()

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
    trnexe = trnpy.core.TRNExe(
                          mode_exec_parallel=True,
                          mode_trnsys_hidden=False,
                          )

    # Run the TRNSYS simulations
    dck_list = trnexe.run_TRNSYS_dck_list(dck_list)

    # Report any errors that occured
    dck_proc.report_errors(dck_list)

    # Post-processing: This is where the example gets quite specific
    def read_file_function(result_file_path):
        '''Different TRNSYS printer outputs are formatted differently. We
        have to read the file in a way that gives us a usable pandas
        DataFrame. In this case the default TRNSYS output is particulary
        nasty! We have to account for tabs and whitespaces in the separator.
        '''
        df = pd.read_csv(result_file_path,
                         sep=r'\t*\s+',
                         engine='python',
                         skiprows=0, header=[0, 1])
        df.columns = df.columns.droplevel(1)  # Skip the row with units
        return df

    # Collect the results of the simulations. Our goal is to combine
    # the result files of the parametric runs into DataFrames.
    # Put the time and parameter columns into the index of the DataFrame.
    data_start_date = '2019-01-01'
    result_data = dck_proc.results_collect(dck_list, read_file_function,
                                           origin=data_start_date)

    # Now we have got all the results in memory. We can call them by their file
    # paths (as assigned in TRNSYS).
    df_hour = result_data[r'Results.txt']
    print(df_hour.head())

    # Slicing the index for a new time interval
    df_hour = dck_proc.results_slice_time(df_hour, '2019-01-01', '2019-02-01')
#    print(df_hour)

    # Now depending on our needs, we can also group the data by week or year
    # For resampling, we need to choose between using the mean (e.g.
    # temperatures) or sum (e.g. energies). Here we use mean for all columns.
    freq = 'D'
#    freq = 'W'
#    freq = 'M'
#    freq = 'Y'
    df_resample = dck_proc.results_resample(df_hour, freq=freq,
                                            regex_mean=r'_',
                                            prio='mean',
                                            )
    print(df_resample)


def trnsys_batch_example_02(dck_file_list):
    # Create a DCK_processor object. It gives us methods to create and
    # manipulate the deck files that we want to work with
    dck_proc = trnpy.core.DCK_processor()

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
    trnexe = trnpy.core.TRNExe(
                          mode_exec_parallel=True,
                          mode_trnsys_hidden=True,
                          n_cores=7,
                          )

    # Run the TRNSYS simulations
    dck_list = trnexe.run_TRNSYS_dck_list(dck_list)


if __name__ == "__main__":
    '''This function is executed when the script is started directly with
    Python, not when it is loaded as a module.
    '''
    main()
