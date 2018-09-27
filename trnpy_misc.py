# -*- coding: utf-8 -*-
'''
@author: Joris Nettelstroth

TRNpy: Parallelized TRNSYS simulation with Python
=================================================
Simulate TRNSYS deck files in serial or parallel and use parametric tables to
perform simulations for different sets of parameters. TRNpy helps to automate
these and similar operations by providing functions to manipulate deck files
and run TRNSYS simulations from a programmatic level.

TRNpy_misc is a collection of miscellaneous functions that are often useful
when working with TRNpy. These are often wrappers around functions provided
by other modules such as Pandas, Bokeh, Scikit-Optimize and DataExplorer.
'''

import os
import logging
import multiprocessing
import matplotlib.pyplot as plt  # Plotting library
import pandas as pd
import yaml
from bokeh.command.bootstrap import main
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Spectral11 as palette_default

# Define the logging function
logger = logging.getLogger(__name__)


def df_to_excel(df, path, sheet_names=[], merge_cells=False, **kwargs):
    '''Wrapper around pandas' function ``DataFrame.to_excel()``, which creates
    the required directory and only logs any errors instead of terminating
    the script (happens often when the Excel file is currently opended).
    Additional keyword arguments are passed down to ``to_excel()``.
    Can save a single DataFrame to a single Excel file or multiple DataFrames
    to a combined Excel file.


    Args:
        df (DataFrame or list): Pandas DataFrame object(s) to save

        path (str): The full file path to save the DataFrame to

        sheet_names (list): List of sheet names to use when saving multiple
        DataFrames to the same Excel file

        merge_cells (boolean, optional): Write MultiIndex and Hierarchical
        Rows as merged cells. Default False.

        freeze_panes (tuple or boolean, optional): Per default, the sheet
        cells are frozen to always keep the index visible (by determining the
        correct coordinate ``tuple``). Use ``False`` to disable this.

    Returns:
        None
    '''
    import collections

    if not os.path.exists(os.path.dirname(path)):
        logging.debug('Create directory ' + os.path.dirname(path))
        os.makedirs(os.path.dirname(path))

    try:
        if isinstance(df, collections.Sequence) and not isinstance(df, str):
            # Save a list of DataFrame objects into a single Excel file
            writer = pd.ExcelWriter(path)
            for i, df_ in enumerate(df):
                try:  # Use given sheet name, or just an enumeration
                    sheet = sheet_names[i]
                except IndexError:
                    sheet = str(i)
                # Add current sheet to the ExcelWriter by calling this
                # function recursively
                df_to_excel(df=df_, path=writer, sheet_name=sheet,
                            merge_cells=merge_cells, **kwargs)
            writer.save()  # Save the actual Excel file

        else:
            # Per default, the sheet cells are frozen to keep the index visible
            if 'freeze_panes' not in kwargs or kwargs['freeze_panes'] is True:
                # Find the right cell to freeze in the Excel sheet
                kwargs['freeze_panes'] = (1, len(df.index.names))
            elif kwargs['freeze_panes'] is False:
                del(kwargs['freeze_panes'])

            # Save one DataFrame to one Excel file
            df.to_excel(path, merge_cells=merge_cells, **kwargs)

    except Exception as ex:
        logging.exception(ex)
        pass


def df_set_filtered_to_NaN(df, filters, mask, value=float('NaN')):
    '''Set all rows defined by the ``mask`` of the columns that are
    included in the ``filters`` to ``NaN`` (or to the optional ``value``).

    This is useful for filtering out time steps where a component is not
    active from mean values calculated later.
    '''
    for column in df.columns:
        for filter_ in filters:
            if filter_ in column:
                df[column][mask] = float('NaN')
                break  # Break inner for-loop if one filter was matched
    return df


def bokeh_stacked_vbar(df_in, stack_labels, stack_labels_neg=[], tips_cols=[],
                       palette=palette_default, **kwargs):
    '''Create stacked vertical bar plot in Bokeh from TRNSYS results.

    The x-axis will have two groups: hash and TIME.

    Use ``**kwargs`` to pass additional keyword arguments to ``figure()`` like
    ``plot_width``, etc.
    '''

    # Prepare Data
    df = df_in.reset_index()  # Remove index

    df[stack_labels_neg] = df[stack_labels_neg] * -1

    group_names = ['hash', 'TIME']
    for col in group_names:
        if col == 'hash':  # Add leading zero for correct string sorting
            df[col] = [str(df[col][i]).zfill(2) for i in df[col].index]
        else:
            df[col] = df[col].astype(str)  # The axis label needs strings

    df.set_index(group_names, inplace=True)  # Rebuild index
    group = df.groupby(level=group_names)
    source = ColumnDataSource(data=df)

    # Create Plot
    p = figure(x_range=group, **kwargs)
    x_sel = source.column_names[-1]  # An artificial column 'hash_TIME'

    r_pos = p.vbar_stack(stack_labels, x=x_sel, width=1, source=source,
                         color=palette[0:len(stack_labels)],
                         name=stack_labels,
                         legend=[x+" " for x in stack_labels],
                         )
    r_neg = p.vbar_stack(stack_labels_neg, x=x_sel, width=1, source=source,
                         color=palette[-len(stack_labels_neg)],
                         name=stack_labels_neg,
                         legend=[x+" " for x in stack_labels_neg],
                         )

    # Create HoverTool
    tips_list = [(col, "@{"+col+"}") for col in tips_cols]
    for r in r_pos+r_neg:
        label = r.name
        hover = HoverTool(tooltips=[(label, "@{"+label+"}"), *tips_list],
                          renderers=[r])
        p.add_tools(hover)
    # Bokeh 0.13.0: Use $name field
#    tips_list = [(col, "@{"+col+"}") for col in param_table.columns]
#    hover = HoverTool(tooltips=[("$name ", "@$name"), *tips_list])
#    p.add_tools(hover)

    p.xaxis.major_label_orientation = 1.2
    p.legend.background_fill_alpha = 0.5
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    return p


def bokeh_circles_from_df(df_in, x_col, y_cols=[], tips_cols=[], size=10,
                          palette=palette_default, **kwargs):
    '''Create a simple circle plot with Bokeh, where one column ``x_col``
    is plotted against all other columns (or the list ``y_cols``) of a
    Pandas DataFrame.

    Use ``**kwargs`` to pass additional keyword arguments to ``figure()`` like
    ``plot_width``, etc.
    '''
    p = figure(**kwargs)
    df = df_in.reset_index()  # Remove index
    source = ColumnDataSource(data=df)

    if len(y_cols) == 0:  # Per default, use all columns in the DataFrame
        y_cols = df_in.columns

    r_list = []
    for i, y_col in enumerate(y_cols):
        r = p.circle(x_col, y_col, source=source, legend=y_col+' ',
                     color=palette[i], name=y_col, size=size)
        r_list.append(r)
    p.legend.click_policy = 'hide'
    p.xaxis.axis_label = x_col

    # Create HoverTool
    tips_list = [(col, "@{"+col+"}") for col in tips_cols]
    for r in r_list:
        label = r.name
        hover = HoverTool(tooltips=[(label, "@{"+label+"}"), *tips_list],
                          renderers=[r])
        p.add_tools(hover)
    return p


def DataExplorer_mark_index(df):
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


def DataExplorer_open(DatEx_df, data_name='TRNSYS Results', port=80,
                      bokeh_app=r'C:\Users\nettelstroth\Documents' +
                                r'\07 Python\dataexplorer',
                      show=True, output_backend='webgl', mark_index=True):
    '''Open the given DataFrame in the DataExplorer application. TRNpy and
    DataExplorer are a great combination, because the values of parametric
    runs can be viewed and filtered as classes in the DataExplorer.
    '''
    # Mark index column names as classifications
    if mark_index:
        DatEx_df = DataExplorer_mark_index(DatEx_df)

    # Prepare settings:
    data_file = os.path.join(bokeh_app, 'upload', data_name + '.xlsx')
    logger.info('Saving file for DataExplorer... ')
    logger.info(data_file)
    if logger.isEnabledFor(logging.INFO):
        print(DatEx_df.head())

    # Save this as a file that DataExplorer will load again
    DatEx_df.to_excel(data_file, merge_cells=False)

    logger.info('Starting DataExplorer...')

    port_blocked = True
    while port_blocked:
        call_list = ["bokeh", "serve", bokeh_app, "--port", str(port)]
        if show:
            call_list.append("--show")
        call_list += ["--args",
                      "--name", data_name,
                      "--file", data_file,
                      "--bokeh_output_backend", output_backend]
        try:
            main(call_list)  # Call Bokeh app
        except SystemExit:
            # Error would produce ugly print (when port is already in use)
            port += 1  # Increment port number for the next try
        else:  # try was successful, no SystemExit was raised
            port_blocked = False


def skopt_optimize(eval_func, opt_dimensions, n_calls=100, n_cores=0,
                   tol=0.001, random_state=1, plots_show=False,
                   plots_dir=r'.\Result', load_optimizer_pickle_file=None,
                   kill_file='optimizer.yaml', **skopt_kwargs):
    '''Perform optimization for a TRNSYS-Simulation with scikit-optimize.
    https://scikit-optimize.github.io/#skopt.Optimizer

    The "ask and tell" API of scikit-optimize exposes functionality that
    allows to obtain multiple points for evaluation in parallel. Intended
    usage of this interface is as follows:
        1. Initialize instance of the Optimizer class from skopt
        2. Obtain n points for evaluation in parallel by calling the ask
           method of an optimizer instance with the n_points argument set
           to n > 0
        3. Evaluate points
        4. Provide points and corresponding objectives using the tell
           method of an optimizer instance
        5. Continue from step 2 until eg maximum number of evaluations
           reached

    Description copied from here, where more info can be found:
    https://scikit-optimize.github.io/notebooks/parallel-optimization.html

    This function implements the description above, except for step "3.
    Evaluate points". A function ``eval_func`` for this has to be
    provided by the user.

    Args:
        eval_func (function): Evaluation function. Must take a
        ``param_table`` as input, perform TRNSYS simulations, read
        simulation results, and return a list of function results,
        which are to be minimized.

        opt_dimensions (dict): Dictionary with pairs of parameter name
        (as defined in TRNSYS deck) and space dimensions (boundaries as
        defined in skopt.Optimizer, typically a ``(lower_bound,
        upper_bound)`` tuple).

        n_calls (int, default=100): Maximum number of calls to
        ``eval_func``.

        n_cores (int, optional): Number of CPU cores to use in parallel,
        defaults to total number of cores minus 1.

        random_state (int or None, optional):
        Set random state to something other than None for reproducible
        results. Default = 1.

        plots_show (bool, optional): Show evaluations and dimensions
        plots provided by skopt.plots. Default = False.

        plots_dir (str, optional): Directory to save skopt.plots into. If
        ``None``, no plots are saved. Default is ``".\Result"``

        load_optimizer_pickle_file (str, optional): A path to an optimizer
        instance dumped before with pickle. This allows to continue a
        previous optimization process. Default is ``None``.

        kill_file (str, optional): A path to a yaml file. If it contains
        the entry ``kill: True``, the optimization is stopped before the next
        round. Default is ``"optimizer.yaml"``

        skop_kwargs: Optional keyword arguments that are passed on to
        skopt.Optimizer, e.g.

            * n_initial_points (int, default=10):
              Number of evaluations of `func` with random initialization
              points before approximating it with `base_estimator`.

            * See more: https://scikit-optimize.github.io/#skopt.Optimizer

    Returns:
        opt_res (OptimizeResult, scipy object): The optimization result
        returned as a ``OptimizeResult`` object.
    '''
    import skopt
    from skopt.plots import plot_evaluations, plot_objective
    import pickle

    if n_cores == 0:  # Set number of CPU cores to use
        n_cores = multiprocessing.cpu_count() - 1

    if load_optimizer_pickle_file is not None:
        # Load an existing optimizer instance
        with open(load_optimizer_pickle_file, 'rb') as f:
            sk_optimizer = pickle.load(f)
            logger.info('Optimizer: Loaded existing optimizer instance '
                        + load_optimizer_pickle_file)
    else:
        # Default behaviour: Start fresh with a new optimizer instance
        sk_optimizer = skopt.Optimizer(
            dimensions=opt_dimensions.values(),
            random_state=random_state,
            **skopt_kwargs,
        )

    # Start the optimization loop
    for count in range(1, n_calls+1):
        logger.info('Optimizer: Starting iteration round '+str(count))
        try:
            next_x = sk_optimizer.ask(n_points=n_cores)  # points to evaluate
        except ValueError as ex:
            logger.exception(ex)
            continue
        param_table = pd.DataFrame.from_records(
                next_x, columns=opt_dimensions.keys())
        next_y = eval_func(param_table)  # evaluate points in parallel
        result = sk_optimizer.tell(next_x, next_y)
        result.nit = count*n_cores
        result.labels = list(opt_dimensions.keys())

        if result.fun < tol:
            result.success = True
            break
        else:
            result.success = False

        # Save intermediate results after each round as pickle objects
        if plots_dir is not None and not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        if plots_dir is not None:
            with open(os.path.join(plots_dir, 'optimizer.pkl'), 'wb') as f:
                pickle.dump(sk_optimizer, f)
            with open(os.path.join(plots_dir, 'opt_result.pkl'), 'wb') as f:
                pickle.dump(result, f)

        # A yaml file in the current working directory allows to stop
        # the optimization and proceed with the program
        try:
            kill_dict = yaml.load(open(kill_file, 'r'))
            if kill_dict.get('kill', False):
                logger.critical('Optimizer: Killed by file '+kill_file)
                kill_dict['kill'] = False
                yaml.dump(kill_dict, open(kill_file, 'w'),
                          default_flow_style=False)
                break
        except Exception:
            pass

    logger.info('Optimizer: Best fit after '+str(count)+' rounds: '
                + str(result.fun))

    # Generate, show and save optimization result plots:
    if result.space.n_dims > 1:
        plots_dir = os.path.abspath(plots_dir)
        if plots_dir is not None and not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        skopt.plots.plot_evaluations(result, dimensions=result.labels)
        if plots_dir is not None:
            plt.savefig(os.path.join(plots_dir, r'skopt_evaluations.png'),
                        bbox_inches='tight')
        try:  # plot_objective might fail
            skopt.plots.plot_objective(result, dimensions=result.labels)
        except IndexError as ex:
            logger.error('Error "' + str(ex) + '". Probably not enough ' +
                         'data to plot partial dependence')
        else:
            if plots_dir is not None:
                plt.savefig(os.path.join(plots_dir, r'skopt_objective.png'),
                            bbox_inches='tight')

        if plots_show is True:
            plt.show()

    return result


if __name__ == "__main__":
    '''This is executed when the script is started directly with
    Python, not when it is loaded as a module.
    '''
    # Define output format of logging function
    logging.basicConfig(format='%(asctime)-15s %(message)s')
