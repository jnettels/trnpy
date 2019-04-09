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

Module Misc
-----------
This is a collection of miscellaneous functions that are often useful
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
from bokeh.models import HoverTool, ColumnDataSource, RangeTool
from bokeh.layouts import column
from bokeh.palettes import Spectral11 as palette_default

# Define the logging function
logger = logging.getLogger(__name__)


def df_to_excel(df, path, sheet_names=[], merge_cells=False,
                check_permission=True, **kwargs):
    '''Wrapper around pandas' function ``DataFrame.to_excel()``, which creates
    the required directory.
    In case of a ``PermissionError`` (happens when the Excel file is currently
    opended), the file is instead saved with a time stamp.

    Additional keyword arguments are passed down to ``to_excel()``.
    Can save a single DataFrame to a single Excel file or multiple DataFrames
    to a combined Excel file.

    The function calls itself recursively to achieve those features.

    Args:
        df (DataFrame or list): Pandas DataFrame object(s) to save

        path (str): The full file path to save the DataFrame to

        sheet_names (list): List of sheet names to use when saving multiple
        DataFrames to the same Excel file

        merge_cells (boolean, optional): Write MultiIndex and Hierarchical
        Rows as merged cells. Default False.

        check_permission (boolean): If the file already exists, instead try
        to save with an appended time stamp.

        freeze_panes (tuple or boolean, optional): Per default, the sheet
        cells are frozen to always keep the index visible (by determining the
        correct coordinate ``tuple``). Use ``False`` to disable this.

    Returns:
        None
    '''
    from collections.abc import Sequence
    import time

    if check_permission:
        try:
            # Try to complete the function without this permission check
            df_to_excel(df, path, sheet_names=sheet_names,
                        merge_cells=merge_cells, check_permission=False,
                        **kwargs)
            return  # Do not run the rest of the function
        except PermissionError as e:
            # If a PermissionError occurs, run the whole function again, but
            # with another file path (with appended time stamp)
            logger.critical(e)
            ts = time.localtime()
            ts = time.strftime('%Y-%m-%d_%H-%M-%S', ts)
            path_time = (os.path.splitext(path)[0] + '_' +
                         ts + os.path.splitext(path)[1])
            logger.critical('Writing instead to:  '+path_time)
            df_to_excel(df, path_time, sheet_names=sheet_names,
                        merge_cells=merge_cells, **kwargs)
            return  # Do not run the rest of the function

    # Here the 'actual' function content starts:
    if not os.path.exists(os.path.dirname(path)):
        logging.debug('Create directory ' + os.path.dirname(path))
        os.makedirs(os.path.dirname(path))

    if isinstance(df, Sequence) and not isinstance(df, str):
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
            if merge_cells:
                freeze_rows = len(df.columns.names) + 1
            else:
                freeze_rows = 1

            kwargs['freeze_panes'] = (freeze_rows, len(df.index.names))
        elif kwargs['freeze_panes'] is False:
            del(kwargs['freeze_panes'])

        # Save one DataFrame to one Excel file
        df.to_excel(path, merge_cells=merge_cells, **kwargs)


def df_set_filtered_to_NaN(df, filters, mask, value=float('NaN')):
    '''Set all rows defined by the ``mask`` of the columns that are
    included in the ``filters`` to ``NaN`` (or to the optional ``value``).

    This is useful for filtering out time steps where a component is not
    active from mean values calculated later.
    '''
    for column_ in df.columns:
        for filter_ in filters:
            if filter_ in column_:
                df[column_][mask] = float('NaN')
                break  # Break inner for-loop if one filter was matched
    return df


def bokeh_stacked_vbar(df_in, stack_labels, stack_labels_neg=[], tips_cols=[],
                       palette=palette_default, y_label=None, **kwargs):
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
    x_sel = source.column_names[0]  # An artificial column 'hash_TIME'

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
    p.toolbar.autohide = True
    if y_label is not None:
        p.yaxis.axis_label = y_label
    return p


def bokeh_circles_from_df(df_in, x_col, y_cols=[], tips_cols=[], size=10,
                          palette=palette_default, **kwargs):
    '''Create a simple circle plot with Bokeh, where one column ``x_col``
    is plotted against all other columns (or the list ``y_cols``) of a
    Pandas DataFrame.

    Use ``**kwargs`` to pass additional keyword arguments to ``figure()`` like
    ``plot_width``, etc.
    '''
    if len(y_cols) == 0:  # Per default, use all columns in the DataFrame
        y_cols = list(df_in.columns)

    df = df_in.reset_index()  # Remove index
    selection = y_cols + [x_col] + list(tips_cols)
    source = ColumnDataSource(data=df[selection])  # Use required columns

    r_list = []
    p = figure(**kwargs)
    for i, y_col in enumerate(y_cols):
        r = p.circle(x_col, y_col, source=source, legend=y_col+' ',
                     color=palette[i], name=y_col, size=size)
        r_list.append(r)
    p.legend.click_policy = 'hide'
    p.xaxis.axis_label = x_col
    p.toolbar.autohide = True

    # Create HoverTool
    tips_list = [(col, "@{"+col+"}") for col in tips_cols]
    for r in r_list:
        label = r.name
        hover = HoverTool(tooltips=[(label, "@{"+label+"}"), *tips_list],
                          renderers=[r])
        p.add_tools(hover)
    return p


def bokeh_time_lines(df, fig_link=None, index_level='hash', **kwargs):
    '''Create multiple line plot figures with ``bokeh_time_line()``,
    one for each hash in the DataFrame.

    Args:
        df (DataFrame): Simulation results

        fig_link (bokeh figure): A Bokeh figure that you want to link the
        x-axis to. Usefull if you call ``bokeh_time_lines()`` several times
        with different ``y_cols``.

        index_level (str): Name of the index level for whose values individual
        plots will be created. Default = 'hash'.

    kwargs:
        y_cols (list): List of column names to plot on the y-axis.
        Is passed down to ``bokeh_time_line()``.

        Other keyword arguments are passed to ``bokeh_time_line()``,
        where they are passed to Bokeh's ``figure()``, e.g. ``plot_width``.

    Return:
        A list of the Bokeh figures.
    '''

    fig_list = []  # List of Bokeh figure objects (Sankey plots)
    for hash_ in set(df.index.get_level_values(index_level)):
        df_plot = df.loc[(hash_, slice(None), slice(None)), :]

        title = []
        for j, level in enumerate(df_plot.index.names):
            if level == 'TIME':
                    continue
            label = df_plot.index.codes[j][0]
            title += [level+'='+str(df_plot.index.levels[j][label])]

        if len(fig_list) == 0 and fig_link is None:
            col = bokeh_time_line(df_plot, **kwargs,
                                  title=', '.join(title))
        elif fig_link is not None:  # Use external figure for x_range link
            col = bokeh_time_line(df_plot, **kwargs,
                                  title=', '.join(title),
                                  fig_link=fig_link)
        else:  # Give first figure as input to other figures x_range link
            col = bokeh_time_line(df_plot, **kwargs,
                                  title=', '.join(title),
                                  fig_link=fig_list[0].children[0])

        fig_list += [col]

    for col in fig_list:  # Link all the y_ranges
        col.children[0].y_range = fig_list[0].children[0].y_range  # figure
        col.children[1].y_range = fig_list[0].children[0].y_range  # range

    return fig_list


def bokeh_time_line(df_in, y_cols=[], palette=palette_default,
                    fig_link=None, y_label=None, x_col='TIME', **kwargs):
    '''Create line plots over a time axis for all or selected columns
    in a DataFrame. A RangeTool is placed below the figure for easier
    navigation.

    Args:
        df_in (DataFrame): Simulation results.

        y_cols (list, optional): List of column names to plot on the y-axis.
        Per default, use all columns in the DataFrame.

        palette (list, optional): List of colours in hex format.

        fig_link (bokeh figure, optional): A Bokeh figure that you want to
        link the x-axis to.

    kwargs:
        Other keyword arguments are passed to Bokeh's ``figure()``,
        e.g. ``plot_width``.

    Returns:
        Column of two figures: The time line plot and a select plot below

    '''
    if len(y_cols) == 0:  # Per default, use all columns in the DataFrame
        y_cols = list(df_in.columns)

    df = df_in.reset_index()  # Remove index
    source = ColumnDataSource(data=df[[x_col]+y_cols])  # Use required columns

    if fig_link is None:
        fig_x_range = (df[x_col].min(), df[x_col].max())
    else:
        fig_x_range = fig_link.x_range

    p = figure(**kwargs, x_axis_type='datetime',
               x_range=fig_x_range)

    for y_col, color in zip(y_cols, palette):
        p.line(x_col, y_col, legend=y_col+' ', line_width=2,
               source=source, color=color, name=y_col)

    hover = HoverTool(tooltips='$name: @$name')
    p.add_tools(hover)
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = '8pt'
    p.legend.spacing = 1
    p.legend.padding = 5

    p.yaxis.major_label_orientation = "vertical"
    if y_label is not None:
        p.yaxis.axis_label = y_label

    # Add a new figure that uses the range_tool to control the figure p
    select = figure(plot_height=45, plot_width=p.plot_width, y_range=p.y_range,
                    x_axis_type="datetime", y_axis_type=None, tools="",
                    toolbar_location=None, background_fill_color="#efefef")

    range_tool = RangeTool(x_range=p.x_range)  # Link figure and RangeTool
    range_tool.overlay.fill_color = palette[0]
    range_tool.overlay.fill_alpha = 0.5

    for y_col in y_cols:
        select.line(x_col, y_col, source=source)

    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool

    return column(p, select)


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

        tol (float, optional): Error tolerance. Optimization finishes with
        ``success`` if a result lower than ``tol`` is achieved.

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

        # Generate and save optimization result plots:
        if result.space.n_dims > 1:
            try:
                plt.close('all')
            except Exception:
                pass

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

    if result.space.n_dims > 1:
        if plots_show is True:
            # Show optimization result plots
            plt.show()

    return result


if __name__ == "__main__":
    '''This is executed when the script is started directly with
    Python, not when it is loaded as a module.
    '''
    # Define output format of logging function
    logging.basicConfig(format='%(asctime)-15s %(message)s')
