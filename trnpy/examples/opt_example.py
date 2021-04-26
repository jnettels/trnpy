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

Module Optimization Example
---------------------------
Provide examples for the use of optimization.

For an optimization to work, you need not only to automate the simulations,
but also the evaluations of the results. TRNpy can take care of the former,
but the latter requires the user to define and calculate some sort of
evaluation function that can be minimized in the optimization.

This example shows how to define a function that controls the optimization
``TRNpy_optimize_scikit`` and a ``Processor`` class that takes care of
simulating with TRNSYS and calculating some key performance indicators (KPI)
that the optimizer can minimize.

For the official example 'Begin.dck' we search for the simulation time (STOP)
that yields a desired amount of collected energy. Only named variables can
be replaced, so this is the only somewhat plausible replacement we can use
for this example.

Requires scikit-optimize: https://scikit-optimize.github.io/stable/

pip install scikit-optimize
conda install -c conda-forge scikit-optimize

"""

import logging
import pandas as pd
import trnpy.core  # trnpy core classes: TRNExe, DCK_processor
import trnpy.misc  # Miscellaneous helper functions

# Define the logging function
logger = logging.getLogger(__name__)


def main(perform_optimization=True):
    """Run the main function that contains all the steps for this script."""
    setup()  # Perform setup

    dck_file_list = [r'C:\Trnsys17\Examples\Begin\Begin.dck']

    # Create a DCK_processor object
    dck_proc = trnpy.core.DCK_processor(sim_folder=r'C:\Trnsys17\Work\batch')
    # Add another file ending to the regex of result files
    dck_proc.regex_result_files = dck_proc.regex_result_files+r'|\.hst'
    # Create a TRNExe object
    trnexe = trnpy.core.TRNExe(
            path_TRNExe=r'C:\Trnsys17\Exe\TRNExe.exe',
            mode_exec_parallel=True,
            mode_trnsys_hidden=True,
            delay=0.1,
            )

    # Create a regular dck_list without additional parameters
    dck_list = dck_proc.create_dcks_from_file_list(
            dck_file_list, update_dest=True, copy_files=True, )

    proc = Processor(trnexe, dck_proc)

    if perform_optimization:
        # Perform optimization with scikit-optimize
        dck_file = dck_list[0].file_path_dest  # Use changed & copied file
        dck_list, param_table = TRNpy_optimize_scikit(proc, dck_proc, dck_file)

        proc.param_table = param_table

    # Run the TRNSYS simulations again, with the best optimization results
    proc.run_and_evaluate(dck_list, warn=True)

    # Now we can access the results stored in our 'proc' object
    print(proc.KPI)


def TRNpy_optimize_scikit(proc, dck_proc, dck_file):
    """Run SciKit-Optimize implementation for multidimensional optimization."""

    def eval_func(param_table):
        """Define evaluation function for optimization with a single objective.

        Takes a param_table as input and returns a list of function values
        that are to be minimized.
        """
        dck_list = dck_proc.parametric_table_auto(param_table, dck_file)
        proc.run_and_evaluate(dck_list, warn=False)

        # Reduce param_table and KPI to the same index, so they can be
        # concatenated
        KPI_copy = proc.KPI.reset_index()
        KPI_copy.set_index('hash', inplace=True)
        param_table.index.name = 'hash'
        eval_df = pd.concat([param_table, KPI_copy[target_col]], axis=1)
        # Calculate the error for each simulation run
        eval_df['set'] = target_val
        eval_df['error'] = abs(eval_df[target_col] - eval_df['set'])

        # If any simulation did not finish, the error is currently 'NaN'.
        # Scikit-Optimize cannot deal with that, so we replace it.
        print(eval_df)
        eval_df.fillna(value=NaN_val, inplace=True)
        error = list(eval_df['error'])
        return error

    target_col = 'QColl'  # Power of solar collector in TRNSYS simulation
    target_val = 1e8  # [kJ] Search for this amount of collected energy
    NaN_val = 0.5  # used if the simulation did not succeed

    opt_dimensions = {
        'STOP': [100, 1000],  # [h] Define min and max for simulation time
        'STEP': [1],  # If a list of length=1 is used, a value stays constant
        }

    opt_res = trnpy.misc.skopt_optimize(
        eval_func,
        opt_dimensions,
        tol=1e6,  # tolerance for succesful optimization
        n_initial_points=20,
        # initial_point_generator='random',
        initial_point_generator='grid',
        n_calls=1000,  # maximum number of allowed simulations
        # n_cores=10,
        plots_dir=r'.\Result_opt',
        )
    # print(opt_res)

    opt_res_df = pd.DataFrame(opt_res.x_iters,
                              columns=opt_res.space.dimension_names)
    opt_res_df['error'] = opt_res.func_vals
    opt_res_df.sort_values(by='error', inplace=True)
    trnpy.misc.df_to_excel(opt_res_df, './Result_opt/opt_res.xlsx')
    param_table = opt_res_df.drop(columns=['error']).head(n=5)
    dck_list = dck_proc.parametric_table_auto(param_table, dck_file)
    print(opt_res_df)

    return dck_list, param_table


class Processor():
    """Define a class for processing simulation and results."""

    def __init__(self, trnexe, dck_proc):
        self.trnexe = trnexe
        self.dck_proc = dck_proc
        self.subdck_name = None  # add_subdck_to_dck_list()
        self.dck_list = None  # add_subdck_to_dck_list()

    def read_file_function(self, result_file_path):
        """Define a file reader function.

        Different TRNSYS printer outputs are formatted differently. We
        have to read the file in a way that gives us a usable pandas
        DataFrame. In this case the default TRNSYS output is particulary
        nasty! We have to account for tabs and whitespaces in the separator.
        """
        if result_file_path.endswith('.hst'):
            df = pd.DataFrame()
        else:
            df = pd.read_csv(result_file_path,
                             sep=r'\t*\s+',
                             engine='python',
                             skiprows=0, header=[0, 1])
            df.columns = df.columns.droplevel(1)  # Skip the row with units
        return df

    def run_and_evaluate(self, dck_list, use_previous_results=False,
                         warn=False):
        """Combine the functions run and evaluate."""
        self.run(dck_list, use_previous_results=use_previous_results,
                 warn=warn)
        self.evaluate()

    def run(self, dck_list, use_previous_results=False, warn=False):
        """Run the TRNSYS simulations."""
        if use_previous_results:
            logger.critical('Using previous TRNSYS results!')
        else:
            dck_list = self.trnexe.run_TRNSYS_dck_list(dck_list)

        self.dck_proc.report_errors(dck_list, warn=warn)

        self.dck_list = dck_list
        return dck_list

    def evaluate(self):
        """Evaluate the TRNSYS results."""
        dck_list = self.dck_list

        # Put the time and parameter columns into the index of the DataFrame.
        origin = '2019-01-01'
        result_data = self.dck_proc.results_collect(
                dck_list, self.read_file_function, origin=origin,
                remove_leap_year=False)

        # Now we have got all the results in memory. We can call them by
        # their file paths (as assigned in TRNSYS).
        df_hour = result_data[r'Begin.out']
        df_hour.drop('success', axis=1, inplace=True)
        if logger.isEnabledFor(logging.DEBUG):
            print(df_hour)

        # Now depending on our needs, we can also group the data by week or
        # year. For resampling, we need to choose between using the mean (e.g.
        # temperatures) or sum (e.g. energies).
        # We want to use the sum of the column 'QColl' in our evaluation
        # function, so we have to define regex_sum accordingly.
        freq = 'Y'
        df_resample = self.dck_proc.results_resample(df_hour, freq=freq,
                                                     regex_mean=r'_',
                                                     regex_sum=r'.*',
                                                     prio='sum',
                                                     )
        self.KPI = df_resample[['QColl', 'QAux']]


def setup():
    """Set up Pandas' terminal output, logger and colours."""
    # multiprocessing.freeze_support()  # Required on Windows

    # Global Pandas option for displaying terminal output
    pd.set_option('display.max_columns', 0)
    # Set the number of decimal points for the following terminal output
    pd.set_option('precision', 2)
    # http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    pd.set_option('mode.chained_assignment', None)

    # Define the logging function
    logging.basicConfig(format='%(asctime)-15s %(message)s')

    # Set loggers of imported modules:
    # log_level = 'DEBUG'
    log_level = 'INFO'
    logger.setLevel(level=log_level.upper())  # Logger for this module
    logging.getLogger('trnpy.core').setLevel(level=log_level.upper())
    logging.getLogger('trnpy.misc').setLevel(level=log_level.upper())


if __name__ == "__main__":
    # This is executed when the script is started directly with
    # Python, not when it is loaded as a module.
    main()
