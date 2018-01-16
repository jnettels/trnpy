# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 18:01:41 2018

@author: nettelstroth
"""
import logging
import os
import trnsys
import multiprocessing
import pandas as pd
from bokeh.command.bootstrap import main


def trnsys_batch_example_01(dck_file):
    # Create a DCK_processor object. It gives us methods to create and
    # manipulate the deck files that we want to work with
    dck_proc = trnsys.DCK_processor()

    # Load a parametric table located in the current directory
    param_table_file = r'Parametrics.xlsx'
    param_table = dck_proc.read_parametric_table(param_table_file)

    # Create a deck list from the parameters and the original deck file
    dck_list = dck_proc.get_parametric_dck_list(param_table, dck_file)

    # Apply parametric modifications to list of decks
    dck_proc.rewrite_dcks(dck_list)

    # Copy all files
    dck_proc.copy_assigned_files(dck_list)

    # Create the TRNSYS object
    trnexe = trnsys.TRNExe(
                           mode_exec_parallel=True,
#                           mode_trnsys_hidden=True,
                           )

    # Run TRNSYS simulations
    dck_list = trnexe.run_TRNSYS_dck_list(dck_list)

    # Report any errors that occured
    dck_proc.report_errors(dck_list)

    # Process the results of the simulations. Here our goal is to combine
    # the result files of the parametric runs into DataFrames
    result_data = dict()  # The dict will have one DataFrame for each file name
    for dck in dck_list:
        for result_file in dck.result_files:
            if 'simsum' in result_file:
                # Skip simsum files for now, they are difficult to read
                continue

            # Read the result file into a DataFrame
            result_file_path = os.path.join(os.path.dirname(
                                            dck.file_path_dest),
                                            result_file)
            df_new = dck_proc.read_filetypes(result_file_path)

            # Add the 'hash' and all the key, value pairs to the DataFrame
            df_new['hash'] = [dck.hash]*len(df_new)
            for key, value in dck.replace_dict.items():
                df_new[key] = [value]*len(df_new)

            # Add the DataFrame to the dict of result files
            if result_file in result_data.keys():
                df_old = result_data[result_file]
            else:
                df_old = pd.DataFrame()
            # Append the old and new df, with a new index. Add it to the dict
            df = pd.concat([df_old, df_new], ignore_index=True)
            result_data[result_file] = df

    # We have got all the results in memory. Now we can start manipulating
    # them. Here we make the TIME column a Pandas DateTime object and add
    # the parametric values to the index. Then we resample the hourly data
    # to weekly data.
    df_hourly = result_data[r'Result\temperaturen_1.out']

    # Convert TIME column to float and then to datetime
    df_hourly['TIME'] = [float(string) for string in df_hourly['TIME']]
    df_hourly['TIME'] = pd.to_datetime(df_hourly['TIME'], unit='h',
                                       origin=pd.Timestamp('2017-01-01'))

    df.rename(columns={'A_Koll': '!A_Koll',
                       'Y_DivKoll': '!Y_DivKoll',
                       }, inplace=True)
    df_hourly.set_index(keys=['!A_Koll', '!Y_DivKoll', 'TIME'], inplace=True)

    df_weekly = dck_proc.resample_using_Grouper(df_hourly, freq='W')
    print(df_weekly)

    # With the manipulation completed, we have the option to view the result:
    open_in_dataexplorer(df_weekly)


def trnsys_batch_example_02(dck_file_list):
    # Create a DCK_processor object. It gives us methods to create and
    # manipulate the deck files that we want to work with
    dck_proc = trnsys.DCK_processor()

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
        replace_dict = {'_WP.out"': '.out"',
                        }
        dck_proc.add_replacements(replace_dict, dck)

        # 3) There is a convenience function to disable all plotters
        dck_proc.disable_plotters(dck)

    # Internally, each dck has stored all those modifications.
    # Now we tell him to actually rewrite the content of the deck files
    dck_proc.rewrite_dcks(dck_list)

    # We copy the decks to their new destination:
    # We must copy all the 'assigned' files as defined in the decks,
    # e.g. weather data, load profiles, etc. to the simulation folder
    dck_proc.copy_assigned_files(dck_list)

    trnexe = trnsys.TRNExe(
                           mode_exec_parallel=True,
#                           mode_trnsys_hidden=True,
                           )

    dck_list = trnexe.run_TRNSYS_dck_list(dck_list,
                                          n_cores=2
                                          )


def open_in_dataexplorer(DatEx_df):
    # Prepare settings:
    bokeh_app = r'C:\Users\nettelstroth\Documents\07 Python\dataexplorer'
    DatEx_data_name = 'TRNSYS Results'
    DatEx_file_path = os.path.join(bokeh_app, 'upload',
                                   'excel_text.xlsx')
#                                   'result_test.csv')

    # Save this as a file that DataExplorer will load again
    DatEx_df.to_excel(DatEx_file_path, merge_cells=False)
#    DatEx_df.to_csv(DatEx_file_path, sep=';', index=True)

    # Call Bokeh app:
    main(["bokeh", "serve", bokeh_app, "--show",
          "--log-level", "warning",
          "--args",
          "--name", DatEx_data_name,
          "--file", DatEx_file_path])


if __name__ == "__main__":
    '''Main function
    This function is executed when the script is started directly with
    Python, not when it is loaded as a module.
    '''
    multiprocessing.freeze_support()  # Required on Windows

    # Global Pandas option for displaying terminal output
    pd.set_option('display.max_columns', 0)

    # Define the logging function
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level='DEBUG')
#    logging.basicConfig(format=FORMAT, level='INFO')
#    logging.basicConfig(format=FORMAT, level='ERROR')

    dck_file_list = [
#        r'Steinfurt_180105\Steinfurt_180105.dck',
        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test1.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test2.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test3.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test4.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test5.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test6.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test7.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test8.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test9.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test10.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test11.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test12.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test13.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test14.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test15.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test16.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test17.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test18.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test19.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_171204.dck',
#        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_171201.dck',
#        r'C:\Trnsys17\Work\futureSuN\HK\Hannover_Koll_Multi_170222.dck',
        ]

    # This example takes only a single deck file path as input
    for dck_file in dck_file_list:
        trnsys_batch_example_01(dck_file)

    # This example takes a list of deck file paths as input
#    trnsys_batch_example_02(dck_file_list)
