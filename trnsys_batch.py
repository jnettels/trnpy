# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 18:01:41 2018

@author: nettelstroth
"""
import logging
import os
import trnsys
import multiprocessing
from bokeh.command.bootstrap import main
import pandas as pd

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

    for i in range(1):

        dck_file_list = [
            r'Steinfurt_180105\Steinfurt_180105.dck',
#            r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105.dck',
#            r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test1.dck',
#            r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test2.dck',
#            r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test3.dck',
#            r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test4.dck',
#            r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test5.dck',
#            r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test6.dck',
#            r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test7.dck',
#            r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test8.dck',
#            r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test9.dck',
#            r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test10.dck',
#            r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test11.dck',
#            r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test12.dck',
#            r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test13.dck',
#            r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test14.dck',
#            r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test15.dck',
#            r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test16.dck',
#            r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test17.dck',
#            r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test18.dck',
#            r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_180105_test19.dck',
    #        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_171204.dck',
    #        r'C:\Trnsys17\Work\futureSuN\SB\Steinfurt_171201.dck',
    #        r'C:\Trnsys17\Work\futureSuN\HK\Hannover_Koll_Multi_170222.dck',
            ]
    #    print(dck_list)

#        dck_list = dck_list[:i]

        # Create a DCK_processor object. It gives us methods to create and
        # manipulate the deck files that we want to work with
        dck_proc = trnsys.DCK_processor()

        param_table_file = r'Parametrics.xlsx'
        param_table = dck_proc.read_parametric_table(param_table_file)
        print(param_table)


        for dck_file in dck_file_list:

            dck_list = dck_proc.get_parametric_dck_list(param_table, dck_file)

            dck_proc.rewrite_dcks(dck_list)
#            print(dck_list_new)

#            break
            dck_proc.copy_assigned_files(dck_list)
            trnexe = trnsys.TRNExe(
                                   mode_exec_parallel=True,
#                                   mode_trnsys_hidden=True,
                                   )
        #    print(dck_processor.replace_dict)

            dck_list = trnexe.run_TRNSYS_dck_list(dck_list)


            result_data = dict()
            for dck in dck_list:
                if dck.success == False:
                    print(dck.file_path_dest, dck.error_msg_list)

                for result_file in dck.result_files:
                    if 'simsum' in result_file:
                        continue

                    result_file_path = os.path.join(os.path.dirname(
                                                    dck.file_path_dest),
                                                    result_file)
#                    print(result_file_path)
                    df_new = dck_proc.read_filetypes(result_file_path)
#                    print(result_df.head())
                    df_new['hash'] = [dck.hash]*len(df_new)
                    for key, value in dck.replace_dict.items():
#                        print(dck.hash, key, value)
                        df_new[key] = [value]*len(df_new)

#                    print(result_df.head())
                    if result_file in result_data.keys():
                        df_old = result_data[result_file]
                    else:
                        df_old = pd.DataFrame()
                    # Append (with concatenate) the old and new df, with a new index
                    df = pd.concat([df_old, df_new], ignore_index=True)

                    result_data[result_file] = df

            df_temperature = result_data[r'Result\temperaturen_1.out']
            float_list = []
            for i, string in enumerate(df_temperature['TIME']):
#                string = df_temperature['TIME'][0]
                float_list.append(float(string))
            df_temperature['TIME'] = float_list
#            print(string)
#            fl = float(string)
#            print(fl)
            df_temperature['TIME'] = pd.to_datetime(df_temperature['TIME'],
                          unit='h',
                                      origin=pd.Timestamp('2017-01-01'))
            print(df_temperature['TIME'])

            df.rename(columns={'A_Koll': '!A_Koll',
                               'Y_DivKoll': '!Y_DivKoll',
                               }, inplace=True)
            df_temperature.set_index(keys=['!A_Koll', '!Y_DivKoll', 'TIME'],
                                     inplace=True)

            df_temperature = dck_proc.resample_using_Grouper(df_temperature,
                                                             freq='W')
#                                                             freq='M')
#            df_temperature = df_temperature.resample('M', level='TIME').sum()
            print(df_temperature)
            df_temperature.to_excel('excel_text.xlsx',
                                    merge_cells=False)  # Save this as an Excel file
#            df_temperature.to_csv('result_test.csv',
#                                  sep=';',
#                                  index=True,
#                                  )
#                break

#        break

        if False:
            # Convert the list of file paths to decks into a list of deck objects.
            # These objects will later contain information on the success and
            # results of the simulation
            dck_list = dck_proc.create_dcks_from_file_list(dck_file_list)

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

            # Internally, the dck_processor has stored all those modifications.
            # Now we tell him to actually rewrite the content of the deck files
            dck_proc.rewrite_dcks(dck_list)

            # We copy the decks to their new destination:

            # We must copy all the 'assigned' files as defined in the decks,
            # e.g. weather data, load profiles, etc. to the simulation folder
            dck_proc.copy_assigned_files(dck_list, blacklist=['Result'])

            trnexe = trnsys.TRNExe(
                                   mode_exec_parallel=True,
#                                   mode_trnsys_hidden=True,
                                   )
        #    print(dck_processor.replace_dict)

            return_list = trnexe.run_TRNSYS_dck_list(dck_list,
                                                     n_cores=2
                                                     )


#    if False:
    if True:
        logging.basicConfig(format=FORMAT, level='WARNING')
        bokeh_app = r'C:\Users\nettelstroth\Documents\07 Python\dataexplorer'
        DatEx_data_name = 'TRNSYS Results'
        DatEx_file_path = os.path.abspath(os.path.join(bokeh_app, '..',
                                                       'trnsyspy',
#                                       'result_test.csv'))
                                       'excel_text.xlsx'))
        main(["bokeh", "serve", bokeh_app, "--show", "--args",
              "--name", DatEx_data_name,
              "--file", DatEx_file_path])
