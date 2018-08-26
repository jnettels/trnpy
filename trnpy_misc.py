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
import skopt
from skopt.plots import plot_evaluations, plot_objective
from bokeh.command.bootstrap import main
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Spectral11 as palette_default


def df_to_excel(df, path):
    '''Wrapper around pandas' function DataFrame.to_excel(), which creates
    the required directory and only logs any errors instead of terminating
    the script (happens often when the Excel file is currently opended).

    Args:
        df (DataFrame): Pandas DataFrame object to save

        path (str): The full file path to save the DataFrame to

    Returns:
        None
    '''
    if not os.path.exists(os.path.dirname(path)):
        logging.debug('Create directory ' + os.path.dirname(path))
        os.makedirs(os.path.dirname(path))
    try:
        df.to_excel(path)
    except Exception as ex:
        logging.debug(str(ex))
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
                          palette=palette_default):
    '''Create a simple circle plot with Bokeh, where one column ``x_col``
    is plotted against all other columns (or the list ``y_cols``) of a
    Pandas DataFrame.
    '''
    p = figure()
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
    logging.info('Saving file for DataExplorer... ')
    logging.info(data_file)
    if logging.getLogger().isEnabledFor(logging.INFO):
        print(DatEx_df.head())

    # Save this as a file that DataExplorer will load again
    DatEx_df.to_excel(data_file, merge_cells=False)

    logging.info('Starting DataExplorer...')

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
