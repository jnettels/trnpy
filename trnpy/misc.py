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

Module Misc
-----------
This is a collection of miscellaneous functions that are often useful
when working with TRNpy. These are often wrappers around functions provided
by other modules such as Pandas, Bokeh, Scikit-Optimize and DataExplorer.
"""

import os
import re
import logging
import multiprocessing
import pandas as pd
import yaml
import time
from collections.abc import Sequence
from bokeh.command.bootstrap import main
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool, RangeTool, Panel, Tabs
from bokeh.models.widgets import Div
from bokeh.layouts import layout, column
from bokeh.palettes import Spectral11 as palette_default
from bokeh.palettes import viridis
from bokeh.io import save


# Define the logging function
logger = logging.getLogger(__name__)


def df_to_excel(df, path, sheet_names=[], styles=[], merge_cells=False,
                check_permission=True, **kwargs):
    """Write one or more DataFrames to Excel files.

    Can save a single DataFrame to a single Excel file or multiple DataFrames
    to a combined Excel file with one sheet per DataFrame.
    Is a wrapper around pandas' function ``DataFrame.to_excel()``, which
    creates the required directory.
    In case of a ``PermissionError`` (happens when the Excel file is currently
    opended), the file is instead saved with a time stamp.

    The list ``styles`` allows some basic formatting of specific cells,
    defined by one dict per sheet in the workbook. In the example below,
    a format 'format_EUR' is defined in terms of cell format and width
    and assigned to certain columns:

    .. code:: python

        excel_fmt_eur = '_-* #,##0 €_-;-* #,##0 €_-;_-* "-" €_-;_-@_-'
        styles=[{'columns': {'B:Z': 'format_EUR'},
                 'formats': {'format_EUR': {'num_format': excel_fmt_eur}},
                 'widths': {'format_EUR': 12}}]

    Additional keyword arguments are passed down to ``to_excel()``.

    The function calls itself recursively to achieve those features.

    Args:
        df (DataFrame or list): Pandas DataFrame object(s) to save

        path (str): The full file path to save the DataFrame to

        sheet_names (list, optional): List of sheet names to use when saving
        multiple DataFrames to the same Excel file

        styles (list, optional): List of dicts that contain settings for
        formatting specific cells, one dict per sheet. Dicts must have the
        keys "columns", "formats" and "widths", see example above.

        merge_cells (boolean, optional): Write MultiIndex and Hierarchical
        Rows as merged cells. Default False.

        check_permission (boolean, optional): If the file already exists,
        instead try to save with an appended time stamp.

        freeze_panes (tuple or boolean, optional): Per default, the sheet
        cells are frozen to always keep the index visible (by determining the
        correct coordinate ``tuple``). Use ``False`` to disable this.

    Returns:
        None
    """
    if check_permission:
        try:
            # Try to complete the function without this permission check
            df_to_excel(df, path, sheet_names=sheet_names, styles=styles,
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
            df_to_excel(df, path_time, sheet_names=sheet_names, styles=styles,
                        merge_cells=merge_cells, **kwargs)
            return  # Do not run the rest of the function

    # Here the 'actual' function content starts:
    if not os.path.exists(os.path.dirname(path)):
        logging.debug('Create directory ' + os.path.dirname(path))
        os.makedirs(os.path.dirname(path))

    if isinstance(df, Sequence) and not isinstance(df, str):
        # Save a list of DataFrame objects into a single Excel file
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            for i, df_ in enumerate(df):
                try:  # Use given sheet name, or just an enumeration
                    sheet = str(sheet_names[i])
                except IndexError:
                    sheet = str(i)

                # Add current sheet to the ExcelWriter by calling this
                # function recursively
                df_to_excel(df=df_, path=writer, sheet_name=sheet,
                            merge_cells=merge_cells, **kwargs)

                # Try adding format styles to the workbook sheets
                if len(styles) > 0:
                    try:
                        workbook = writer.book
                        worksheet = writer.sheets[sheet]
                        formats = styles[i]['formats']
                        wb_formats = dict()
                        for fmt_name, format_ in formats.items():
                            # Example: format_ = {'num_format': '0%'}
                            wb_formats[fmt_name] = workbook.add_format(format_)

                        columns = styles[i]['columns']
                        widths = styles[i]['widths']
                        for col, fmt_name in columns.items():
                            # Set the format and width of the column
                            worksheet.set_column(col, widths[fmt_name],
                                                 wb_formats[fmt_name])
                    except Exception as ex:
                        logger.exception(ex)
                        pass

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
    """Set DataFrame values to NaN, based on filters and mask.

    Set all rows defined by the ``mask`` of the columns that are
    included in the ``filters`` to ``NaN`` (or to the optional ``value``).

    This is useful for filtering out time steps where a component is not
    active from mean values calculated later.
    """
    for column_ in df.columns:
        for filter_ in filters:
            if filter_ in column_ or filter_ == column_:
                df[column_][mask] = value
                break  # Break inner for-loop if one filter was matched
    return df


def extract_units_from_df(df, cols_lvl, unit_lvl='Unit'):
    """Extract unit from df columns and return df with MultiIndex.

    Column names must have the form "column_name_unit". The text after the
    last underscore is used as a unit.

    df (DataFrame): The DataFrame with columns to extract units from.

    cols_lvl (str): Name of the level containing the column names with units.
    If columns are not MultiIndex, this name will be given to that level.

    unit_lvl (str, optional): The name of the new level containing the units.


    """
    regex = (r'(?P<name>.+)_(?P<unit>W|kW|MW|kWh|MWh|l|t|kJ|°C|kg/h|%|m/s|°|'
             r'Pa|W/m²|g/kWh|l/s)$')

    tmy2_units = {
        'ETR': 'Wh/m²',
        'ETRN': 'Wh/m²',
        'GHI': 'Wh/m²',
        'DNI': 'Wh/m²',
        'DHI': 'Wh/m²',
        'GHillum': '100 lux',
        'DNillum': '100 lux',
        'DHillum': '100 lux',
        'Zenithlum': '10 Cd/m²',
        'TotCld': 'tenths',
        'OpqCld': 'tenths',
        'DryBulb': '0.1 °C',
        'DewPoint': '0.1 °C',
        'RHum': '%',
        'Pressure': 'mbar',
        'Wdir': '°',
        'Wspd': '0.1 m/s',
        'Hvis': '0.1 km',
        'CeilHgt': 'm',
        'Pwat': 'mm',
        'AOD': '0.001',
        'SnowDepth': 'cm',
        'LastSnowfall': 'days',
        }

    if not isinstance(df.columns, pd.MultiIndex):
        df.columns.set_names(cols_lvl, inplace=True)

    rename_dict = dict()
    unit_list = []

    # Get a match for each column
    for column_ in df.columns.get_level_values(cols_lvl):
        match = re.match(pattern=regex, string=column_)
        if match:
            name = match.group('name')
            unit = match.group('unit')
        elif column_ in tmy2_units.keys():
            name = column_
            unit = tmy2_units[column_]
        else:
            name = column_
            unit = '-'

        rename_dict[column_] = name
        unit_list.append(unit)

    df.rename(columns=rename_dict, level=cols_lvl, inplace=True)
    idx_new = df.columns.to_frame(index=False)
    idx_new[unit_lvl] = unit_list
    df.columns = pd.MultiIndex.from_frame(idx_new)
    return df


def bokeh_stacked_vbar(df_in, stack_labels=[], stack_labels_neg=[],
                       tips_cols=[], palette=palette_default, y_label=None,
                       sum_circle_size=0, **kwargs):
    """Create stacked vertical bar plot in Bokeh from TRNSYS results.

    By default, all columns in the DataFrame will be plotted. Pass ``None``
    to stack_labels and/or stack_labels_neg to prevent that, or pass a list
    with specific column names. If columns in stack_labels_neg are positive,
    they will be made negative.

    If ``sum_circle_size`` is set ``> 0`` a circle with the sum is plotted.

    The x-axis will have two groups: hash and TIME.

    Use ``**kwargs`` to pass additional keyword arguments to ``figure()`` like
    ``plot_width``, etc.
    """
    # Apply logic for default behaviour
    if stack_labels is not None:
        if len(stack_labels) == 0:
            # If no columns are set, use all existing columns
            stack_labels = [c for c in df_in.columns if df_in[c].sum() >= 0]
    else:  # If stack_labels is None, then use an empty list
        stack_labels = []

    if stack_labels_neg is not None:
        if len(stack_labels_neg) == 0:
            # If no columns are set, use all existing columns
            stack_labels_neg = [c for c in df_in.columns if df_in[c].sum() < 0]
    else:  # If stack_labels_neg is None, then use an empty list
        stack_labels_neg = []

    # Filter out non-existing columns
    stack_labels = [c for c in stack_labels if c in df_in.columns]
    stack_labels_neg = [c for c in stack_labels_neg if c in df_in.columns]

    # Filter out empty columns
    stack_labels = [c for c in stack_labels if any(df_in[c] != 0)]
    stack_labels_neg = [c for c in stack_labels_neg if any(df_in[c] != 0)]

    # Prepare Data
    df = df_in.reset_index()  # Remove index

    for col in stack_labels_neg:
        # Make sure values in the 'negative' list are actually negative
        if df[col].sum() > 0:
            df[col] = df[col] * -1
    # Calculate new column 'sum' with sum of each row
    df['sum'] = df[stack_labels].sum(axis=1) + df[stack_labels_neg].sum(axis=1)

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
                         legend_label=[x+" " for x in stack_labels],
                         line_width=0,  # Prevent outline for height of 0
                         )

    if len(p.legend) > 0:
        p.legend[0].items.reverse()  # Reverse order of legend entries

    if len(stack_labels_neg) > 0:
        palette_neg = palette[-len(stack_labels_neg):]
    else:
        palette_neg = []
    r_neg = p.vbar_stack(stack_labels_neg, x=x_sel, width=1, source=source,
                         color=palette_neg, name=stack_labels_neg,
                         legend_label=[x+" " for x in stack_labels_neg],
                         line_width=0,  # Prevent outline for height of 0
                         )
    r_circ = []
    if sum_circle_size > 0:
        r = p.circle(x_sel, 'sum', source=source, legend_label='Sum',
                     name='sum', color=palette[len(stack_labels)+1],
                     size=sum_circle_size)
        r_circ.append(r)

    # Create HoverTool
    tips_list = [(col, "@{"+col+"}") for col in tips_cols]
    for r in r_pos+r_neg+r_circ:
        label = r.name
        hover = HoverTool(tooltips=[(label, "@{"+label+"}"), *tips_list],
                          renderers=[r])
        p.add_tools(hover)
    # Bokeh 0.13.0: Use $name field
    # tips_list = [(col, "@{"+col+"}") for col in param_table.columns]
    # hover = HoverTool(tooltips=[("$name ", "@$name"), *tips_list])
    # p.add_tools(hover)

    p.x_range.range_padding = 0.03  # Extra space on left and right borders
    p.outline_line_color = None
    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = 1.2
    if p.legend:
        p.legend.background_fill_alpha = 0.5
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        # p.toolbar.autohide = True  # TODO Seems bugged
    if y_label is not None:
        p.yaxis.axis_label = y_label
    return p


def bokeh_sorted_load_curve(df, index_level='hash', x_col='TIME', y_label=None,
                            y_cols_line=[], y_cols_stacked=[],
                            palette=palette_default, export_file=False,
                            **kwargs):
    """Create sorted annual load curve.

    The lines can be plotted as is, or stacked.
    """
    from pandas.tseries.frequencies import to_offset

    # Filter out non-existing columns
    y_cols_line = [col for col in y_cols_line if col in df.columns]
    y_cols_stacked = [col for col in y_cols_stacked if col in df.columns]

    # Filter out empty columns (test for NaN and 0)
    y_cols_line = [col for col in y_cols_line if any(df[col].notna())]
    y_cols_stacked = [col for col in y_cols_stacked if any(df[col].notna())]
    y_cols_line = [c for c in y_cols_line if any(df[c] != 0)]
    y_cols_stacked = [c for c in y_cols_stacked if any(df[c] != 0)]

    fig_list = []  # List of Bokeh figure objects
    df_sort_line_list = []
    hash_list = []

    for hash_ in sorted(set(df.index.get_level_values(index_level))):
        df_plot = df.loc[(hash_, slice(None), slice(None)), :]  # use hash only
        df_plot = df_plot.reset_index()  # Remove index
        df_plot.set_index(x_col, inplace=True)  # Make time the only index

        # Create index for x axis of new plot
        freq = pd.to_timedelta(to_offset(pd.infer_freq(df_plot.index)))
        timedelta = df_plot.index[-1] - (df_plot.index[0] - freq)
        index = pd.timedelta_range(start=freq, end=timedelta, freq=freq,
                                   name=x_col) / pd.Timedelta(1, 'h')

        # Create a new DataFrame and fill it with the sorted values
        df_sorted_line = pd.DataFrame(index=index)
        for y_col in y_cols_line:
            sort = df_plot.sort_values(by=y_col, axis=0, ascending=False)
            df_sorted_line[y_col] = sort[y_col].values

        df_sorted_stacked = pd.DataFrame(index=index)
        for y_col in y_cols_stacked:
            sort = df_plot.sort_values(by=y_col, axis=0, ascending=False)
            df_sorted_stacked[y_col] = sort[y_col].values

        df_sorted_stacked.fillna(value=0, inplace=True)
        df_sorted_stacked = df_sorted_stacked.cumsum(axis=1, skipna=False)

        # Create the Bokeh source object used for plotting
        source_line = ColumnDataSource(data=df_sorted_line)
        source_stacked = ColumnDataSource(data=df_sorted_stacked)

        p = figure(title=str(hash_), **kwargs)
        for y_col, color in zip(y_cols_line, palette):
            p.line(x_col, y_col, legend_label=y_col+' ', line_width=2,
                   source=source_line, color=color, name=y_col)

        for y_col, color in zip(y_cols_stacked, palette[len(y_cols_line):]):
            p.line(x_col, y_col, legend_label=y_col+' ', line_width=2,
                   source=source_stacked, color=color, name=y_col)

        hover = HoverTool(tooltips='$name: @$name')
        p.add_tools(hover)
        if p.legend:
            p.legend.location = "top_right"
            p.legend.click_policy = "hide"
            p.legend.label_text_font_size = '8pt'
            p.legend.spacing = 1
            p.legend.padding = 5
            # p.toolbar.autohide = True  # TODO Seems bugged

        if y_label is not None:
            p.yaxis.axis_label = y_label

        fig_list.append(p)
        df_sort_line_list.append(df_sorted_line)
        hash_list.append(hash_)

    if export_file:
        df_to_excel(df=df_sort_line_list, path=export_file,
                    sheet_names=hash_list)

    return fig_list


def bokeh_circles_from_df(df_in, x_col, y_cols=[], tips_cols=[], size=10,
                          palette=palette_default, **kwargs):
    """Create a simple circle plot with Bokeh.

    One column ``x_col`` is plotted against all other columns (or the
    list ``y_cols``) of a Pandas DataFrame.

    Use ``**kwargs`` to pass additional keyword arguments to ``figure()``
    like ``plot_width``, etc.
    """
    if len(y_cols) == 0:  # Per default, use all columns in the DataFrame
        y_cols = list(df_in.columns)

    # Filter out non-existing, empty and NaN columns
    y_cols = [col for col in y_cols if col in df_in.columns]
    y_cols = [col for col in y_cols if any(df_in[col] != 0)]
    y_cols = [col for col in y_cols if any(df_in[col].notna())]

    df = df_in.reset_index()  # Remove index
    selection = y_cols + [x_col] + list(tips_cols)
    source = ColumnDataSource(data=df[selection])  # Use required columns

    r_list = []
    p = figure(**kwargs)
    for i, y_col in enumerate(y_cols):
        r = p.circle(x_col, y_col, source=source, legend_label=y_col+' ',
                     color=palette[i], name=y_col, size=size)
        r_list.append(r)
    p.legend.click_policy = 'hide'
    p.xaxis.axis_label = x_col
    # p.toolbar.autohide = True  # TODO Seems bugged

    # Create HoverTool
    tips_list = [(col, "@{"+col+"}") for col in tips_cols]
    for r in r_list:
        label = r.name
        hover = HoverTool(tooltips=[(label, "@{"+label+"}"), *tips_list],
                          renderers=[r])
        p.add_tools(hover)
    return p


def bokeh_time_lines(df, fig_link=None, index_level='hash', x_col='TIME',
                     **kwargs):
    """Create multiple line plot figures with ``bokeh_time_line()``.

    One plot for each entry in ``index_level`` (hash) in the DataFrame.

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
    """
    fig_list = []  # List of Bokeh figure objects (Sankey plots)
    for hash_ in sorted(set(df.index.get_level_values(index_level))):
        df_plot = df.loc[(hash_, slice(None), slice(None)), :]

        title = []
        for j, level in enumerate(df_plot.index.names):
            if (level == x_col or level == 'TIME'):
                continue
            label = df_plot.index.codes[j][0]
            title += [level+'='+str(df_plot.index.levels[j][label])]

        if len(fig_list) == 0 and fig_link is None:
            col = bokeh_time_line(df_plot, x_col=x_col, **kwargs,
                                  title=', '.join(title))
        elif fig_link is not None:  # Use external figure for x_range link
            col = bokeh_time_line(df_plot, x_col=x_col, **kwargs,
                                  title=', '.join(title),
                                  fig_link=fig_link)
        else:  # Give first figure as input to other figures x_range link
            col = bokeh_time_line(df_plot, x_col=x_col, **kwargs,
                                  title=', '.join(title),
                                  fig_link=fig_list[0].children[0])

        fig_list += [col]

    for col in fig_list:  # Link all the y_ranges
        col.children[0].y_range = fig_list[0].children[0].y_range  # figure
        col.children[1].y_range = fig_list[0].children[0].y_range  # range

    return fig_list


def bokeh_time_line(df_in, y_cols=[], palette=palette_default,
                    fig_link=None, y_label=None, x_col='TIME', **kwargs):
    """Create line plots over a time axis for all or selected columns.

    A RangeTool is placed below the figure for easier navigation.

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

    """
    if len(y_cols) == 0:  # Per default, use all columns in the DataFrame
        y_cols = list(df_in.columns)

    # Filter out non-existing columns
    y_cols = [col for col in y_cols if col in df_in.columns]

    # Filter out empty columns
    y_cols = [col for col in y_cols if any(df_in[col].notna())]

    df = df_in.reset_index()  # Remove index
    source = ColumnDataSource(data=df[[x_col]+y_cols])  # Use required columns

    if fig_link is None:
        fig_x_range = (df[x_col].min(), df[x_col].max())
    else:
        fig_x_range = fig_link.x_range

    if df[x_col].dtype == 'datetime64[ns]':
        x_axis_type = 'datetime'
    else:
        x_axis_type = 'linear'

    p = figure(**kwargs, x_axis_type=x_axis_type, x_range=fig_x_range)

    for y_col, color in zip(y_cols, palette):
        p.line(x_col, y_col, legend_label=y_col+' ', line_width=2,
               source=source, color=color, name=y_col)

    hover = HoverTool(tooltips='$name: @$name')
    p.add_tools(hover)
    if p.legend:
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.legend.label_text_font_size = '8pt'
        p.legend.spacing = 0
        p.legend.padding = 2
    p.title.text_font_size = '8pt'

    p.yaxis.major_label_orientation = "vertical"
    if y_label is not None:
        p.yaxis.axis_label = y_label

    # Add a new figure that uses the range_tool to control the figure p
    select = figure(plot_height=45, plot_width=p.plot_width, y_range=p.y_range,
                    x_axis_type=x_axis_type, y_axis_type=None, tools="",
                    toolbar_location=None, background_fill_color="#efefef",
                    sizing_mode=kwargs.get('sizing_mode', None))

    range_tool = RangeTool(x_range=p.x_range)  # Link figure and RangeTool
    range_tool.overlay.fill_color = palette[0]
    range_tool.overlay.fill_alpha = 0.5

    for y_col in y_cols:
        select.line(x_col, y_col, source=source)

    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool

    return column(p, select)


def create_bokeh_htmls(df_list, files, subfolder='Bokeh', html_show=False,
                       sizing_mode='stretch_width', **kwargs):
    """Create interactive HTML files with Bokeh, for each DataFrame.

    Args:
        df_list (list): List of Pandas DataFrames.

        files (list): List of original filenames.

        subfolder (st, optional): Subfolder for output. Defaults to 'Bokeh'.

        html_show (bool, optional): Automatically open the html file with
        a webbrowser. Defaults to False.

    Returns:
        None.

    """
    for df, file in zip(df_list, files):
        # Construct the output path
        filename = os.path.splitext(os.path.basename(file))[0] + '.html'
        filepath = os.path.join(os.path.dirname(file), subfolder, filename)

        # Create the output folder, if it does not already exist
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        # Create the output file
        logger.info('Saving %s', filepath)
        create_bokeh_html(df, title=os.path.basename(file),
                          html_filename=filepath, html_show=html_show,
                          sizing_mode=sizing_mode, **kwargs)


def create_bokeh_html(df, title='Bokeh', tab_grouper=None,
                      html_filename='bokeh.html',
                      html_show=True, sync_xaxis=True,
                      sizing_mode='stretch_width',
                      time_names=['Zeit', 'Time', 'TIME'],
                      dropna_cols=False,
                      margin=(0, 9, 0, 0),
                      **kwargs):
    """Create a Bokeh html document from a Pandas DataFrame.

    Expects a multiindex in the columns (axis 1), where the levels contain
    the unit and name for each column. The index of axis 0 is the time.

    Args:
        df (DataFrame): Pandas DataFrame with timeseries data to be plotted.

        title (str, optional): Title of the html document. Defaults to 'Bokeh'.

        html_show (bool, optional): True: Open html in browser. False: Only
        save html file to disk. Defaults to True.

        sync_xaxis (bool, optional): Synchronize the x-axis of the plots.
        This is a nice feature, but impacts performance. Defaults to False.

        dropna_cols (bool, optional): Drop columns that are all NaN.

        tab_grouper (str, optional): Name of an index level by which to group
        plots into tabs. If None, do not use tabs.

        **kwargs: Other keyword arguments are handed to Bokeh's figure().

    Returns:
        None.

    """
    if not os.path.exists(os.path.dirname(html_filename)):
        os.makedirs(os.path.dirname(html_filename))

    if isinstance(df.index, pd.MultiIndex):
        # Assume that the first index level is the "hash" that defines
        # different simulations
        tab_list = []
        for _hash in df.index.get_level_values(tab_grouper).unique():
            df_hash = df.xs(_hash, level=tab_grouper)  # Select current hash

            if dropna_cols:
                df_hash = df_hash.copy().dropna(axis='columns', how='all')

            # Extrakt the parameters to create a header title in the html
            title_param_list = []
            names = df_hash.index.names
            parameters = [df_hash.index.get_level_values(n)[0] for n in names]
            for name, param in zip(names, parameters):
                title_param_list.append('{}={}'.format(name, param))
            title_param = ', '.join(title_param_list)
            div = Div(text=title_param)

            # keep only time index
            for time_name in time_names:
                if time_name in df_hash.index.names:
                    time_idx = time_name
            df_hash.index = df_hash.index.get_level_values(time_idx)
            # Get the list of figures for the current hash
            fig_list = create_bokeh_timelines(df_hash, sync_xaxis=sync_xaxis,
                                              sizing_mode=sizing_mode,
                                              margin=margin, **kwargs)
            _column = column([div, *fig_list], sizing_mode=sizing_mode)
            tab_list.append(Panel(child=_column, title=str(_hash)))

        elements = Tabs(tabs=tab_list)
    else:
        elements = create_bokeh_timelines(df, sync_xaxis=sync_xaxis,
                                          sizing_mode=sizing_mode,
                                          **kwargs)

    # Define the layout with all elements
    doc_layout = layout(elements, sizing_mode=sizing_mode)
    # Create the output file
    output_file(html_filename, title=title)
    if html_show:
        show(doc_layout)  # Trigger opening a browser window with the html
    else:
        save(doc_layout)  # Only save the html without showing it


def create_bokeh_timelines(df, sync_xaxis=True, group_lvl=None,
                           unit_lvl=None, sizing_mode='stretch_width',
                           n_cols_max=15, **kwargs):
    """Create Bokeh plots from a Pandas DataFrame.

    Args:
        df (DataFrame): Pandas DataFrame with timeseries data to be plotted.

        sync_xaxis (bool, optional): Synchronize the x-axis of the plots.
        This is a nice feature, but impacts performance. Defaults to False.

        n_cols_max (int, optional): Limit for number of columns to show
        within the same figure. Default: 15

        group_lvl (str, optional): Name of a column level by which to group
        the figures, and use as y-axis label.

        unit_lvl (str, optional): Name of column level that indicates the
        unit of the values in that column. Columns with the same unit are
        grouped into figures and labeled, e.g. [°C]

    Returns:
        None.

    """
    kwargs.setdefault('plot_height', 350)

    # Determine the type of the x axis
    if df.index.inferred_type == 'datetime64':
        x_axis_type = 'datetime'
    else:
        x_axis_type = 'linear'

    # For DataFrame with Multiindex, get a list of unique units
    if group_lvl in df.columns.names:
        groups = list(df.columns.get_level_values(group_lvl).unique())
    else:
        groups = [None]

    if unit_lvl is not None:
        units = list(df.columns.get_level_values(unit_lvl).unique())
    else:
        units = [None]

    # For each DataFrame, create a bokeh figure
    fig_list = []
    for group in groups:
        if group is None:
            df_group = df
        else:
            df_group = df.xs(group, level=group_lvl, axis='columns')

        for unit in units:
            if unit is None:
                df_unit = df_group
            elif unit in df_group.columns.get_level_values(unit_lvl):
                # Make a cross-selection with the current unit
                df_unit = df_group.xs(unit, level=unit_lvl, axis='columns')
            else:
                continue  # Skip this unit
            if len(fig_list) > 0:
                # Synchronize the x_ranges of all subsequent figures to first
                if sync_xaxis:
                    fig_link = fig_list[0].children[0]
            else:
                fig_link = None

            if group is None and unit is None:
                y_label = ""
            elif group is None:
                y_label = unit
            elif unit is None:
                y_label = group
            else:
                y_label = '{} [{}]'.format(group, unit)

            # Do not put more than n_cols_max columns into the same figure
            col_s = 0  # Start column
            col_e = n_cols_max  # End column
            while col_e < len(df_unit.columns)+n_cols_max:
                cols = df_unit.columns[col_s:min(col_e, len(df_unit.columns))]
                col_s = col_e
                col_e += n_cols_max

                p = create_bokeh_timeline(
                    df_unit[cols], fig_link=fig_link, y_label=y_label,
                    x_axis_type=x_axis_type, sizing_mode=sizing_mode,
                    **kwargs)
                fig_list.append(p)

    return fig_list


def create_bokeh_timeline(df, fig_link=None, y_label=None, title=None,
                          output_backend='canvas', x_axis_type='linear',
                          sizing_mode='stretch_both', **kwargs):
    """Create a Bokeh plot from a Pandas DataFrame.

    Args:
        df (DataFrame): Pandas DataFrame with timeseries data to be plotted.

        fig_link (figure, optional): A Bokeh figure to link the x axis with.

        y_label (str, optional): Label for the y axis (e.g. unit).

        title (str, optional): Plot title.

        output_backend (str, optinal): Bokeh's rendering backend
        (``"canvas``", ``"webgl"`` or ``"svg"``).

        x_axis_type (str, optional): Type of x axis, 'datetime' or 'linear'.

    Returns:
        Bokeh 'column' layout object with figures.

    """
    x_col = df.index.name  # Get name of index (used as x-axis)
    if x_col is None:
        logger.debug('No index name found, plot may not show properly!')
        x_col = 'index'  # The default name that ColumnDataSource will produce
    source = ColumnDataSource(df)  # Bokeh's internal data structure

    if fig_link is None:  # Set a default range for the x axis
        fig_x_range = (df.index.min(), df.index.max())
    else:  # link to the range of the given figure
        fig_x_range = fig_link.x_range

    kwargs.setdefault('plot_width', 1000)
    kwargs.setdefault('plot_height', 250)
    # Create the primary plot figure
    p = figure(x_axis_type=x_axis_type,
               x_range=fig_x_range, output_backend=output_backend,
               title=title, **kwargs)

    # Create the glyphs in the figure (one line plot for each data column)
    y_cols = list(df.columns)
    palette = viridis(len(y_cols))  # Generate color palette of correct length
    for y_col, color in zip(y_cols, palette):
        try:
            if not df[y_col].isna().all():
                if isinstance(y_col, tuple):  # if columns have MultiIndex
                    y_col = "_" . join(y_col)  # join to match 'source' object
                r = p.line(x=x_col, y=y_col, source=source, color=color,
                           name=y_col, legend_label=y_col)
                # Add a hover tool to the figure
                add_hover_tool(p, renderers=[r], x_col=x_col,
                               x_axis_type=x_axis_type)
        except ValueError:
            logger.error('Error with column "%s"', y_col)
            raise

    if all([df[y_col].isna().all() for y_col in y_cols]):
        return column([p], sizing_mode=sizing_mode)
    else:
        custom_bokeh_settings(p)  # Set additional features of the plot

        if y_label is not None:
            p.yaxis.axis_label = y_label

        select = get_select_RangeTool(p, x_col, y_cols, source, palette,
                                      output_backend, x_axis_type=x_axis_type)

        return column([p, select], sizing_mode=sizing_mode)


def get_select_RangeTool(p, x_col, y_cols, source, palette=viridis(1),
                         output_backend='canvas', x_axis_type='datetime'):
    """Return a new figure that uses the RangeTool to control the figure p.

    Args:
        p (figure): Bokeh figure to control with RangeTool.

        x_col (str): Name of column for x axis.

        y_cols (list): Names of columns for y axis.

        source (ColumnDataSource): Bokeh's data.

        palette (list, optional): Color palette. Defaults to viridis(1).

        output_backend (str, optional): Bokeh's rendering backend. Defaults
        to 'canvas'.

        x_axis_type (str, optional): Type of x axis, 'datetime' or 'linear'.

    Returns:
        select (figure): Bokeh figure with a range tool.

    """
    select = figure(plot_height=45, y_range=p.y_range, tools="",
                    x_axis_type=x_axis_type, y_axis_type=None,
                    toolbar_location=None, background_fill_color="#efefef",
                    output_backend=output_backend,
                    height_policy="fixed", width_policy="fit",
                    )
    for y_col in y_cols:  # Show all lines of primary figure in "select", too
        if isinstance(y_col, tuple):  # if columns have MultiIndex
            y_col = "_" . join(y_col)  # join to match 'source' object
        select.line(x=x_col, y=y_col, source=source)

    # Create a RangeTool, that will be applied to the "select" figure
    range_tool = RangeTool(x_range=p.x_range)  # Link figure and RangeTool
    range_tool.overlay.fill_color = palette[0]
    range_tool.overlay.fill_alpha = 0.25

    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool
    return select


def custom_bokeh_settings(p):
    """Define common settings for a Bokeh figure p."""
    p.outline_line_color = None
    p.xgrid.grid_line_color = None
    p.yaxis.major_label_orientation = "vertical"
    # p.xaxis.major_label_orientation = 1.2
    if p.legend:
        # p.legend.background_fill_alpha = 0.5
        # p.legend.location = "top_left"
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"  # clickable legend items
        p.legend.spacing = -3
        p.legend.padding = 2
        p.legend.label_text_font_size = '9pt'
    p.toolbar.logo = None  # Remove Bokeh logo


def add_hover_tool(p, renderers, x_col='TIME', x_axis_type='datetime'):
    """Add hover tool to a figure p.

    https://docs.bokeh.org/en/latest/docs/user_guide/tools.html#basic-tooltips
    """
    for r in renderers:
        label = r.name
        if x_axis_type == 'datetime':
            # This allows to add the x_col as a datetime
            tooltips = [(label, "@{"+label+"}"),
                        (x_col, '@{'+x_col+'}{%Y-%m-%d %H:%M %Z}')]
            formatters = {'@{'+x_col+'}': 'datetime'}
        else:
            tooltips = [(label, "@{"+label+"}"),
                        (x_col, '@{'+x_col+'}')]
            formatters = {}

        hover = HoverTool(tooltips=tooltips,
                          renderers=[r],
                          formatters=formatters,
                          # mode='vline',  # is irritating with many lines
                          )
        p.add_tools(hover)



def DataExplorer_mark_index(df):
    """Put '!' in front of index column names.

    For the program ``DataExplorer`` this marks them as classifications.
    Is not applied to time columns.
    """
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
    """Open the given DataFrame in the DataExplorer application.

    TRNpy and DataExplorer are a great combination, because the values of
    parametric runs can be viewed and filtered as classes in the DataExplorer.
    """
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
                   plots_dir=r'.\Result', yscale='linear',
                   optimizer_pickle=None, opt_cfg='optimizer.yaml',
                   **skopt_kwargs):
    r"""Perform optimization for a TRNSYS-Simulation with scikit-optimize.

    https://scikit-optimize.github.io/stable/modules/generated/skopt.Optimizer.html#skopt.Optimizer

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

    More important information:
    https://scikit-optimize.github.io/stable/auto_examples/exploration-vs-exploitation.html#sphx-glr-auto-examples-exploration-vs-exploitation-py

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

        yscale (str, optional): y-axis scale for convergence plot. Options
        are "linear", "log", "symlog", "logit". Default is 'linear'

        optimizer_pickle (str, optional): A path to an optimizer
        instance dumped before with pickle. This allows to continue a
        previous optimization process. Default is ``None``.

        opt_cfg (str, optional): A path to a yaml file. If it contains
        the entry ``kill: True``, the optimization is stopped before the next
        round. Default is ``"optimizer.yaml"``

        skopt_kwargs: Optional keyword arguments that are passed on to
        skopt.Optimizer, e.g.

            * n_initial_points (int, default=10):
              Number of evaluations of `func` with random initialization
              points before approximating it with `base_estimator`.

            * initial_point_generator (str, default: `"random"`):
              Sets a initial points generator. Can be either

              - `"random"` for uniform random numbers,
              - `"sobol"` for a Sobol sequence,
              - `"halton"` for a Halton sequence,
              - `"hammersly"` for a Hammersly sequence,
              - `"lhs"` for a latin hypercube sequence,
              - `"grid"` for a uniform grid sequence

            * See more: https://scikit-optimize.github.io/stable/modules/generated/skopt.Optimizer.html#skopt.Optimizer

    Returns:
        opt_res (OptimizeResult, scipy object): The optimization result
        returned as a ``OptimizeResult`` object.


    .. note::

        I made the following change to ``skopt\optimizer\optimizer.py``
        in function ``def _ask(self)`` to prevent dublicate evaluations:

        .. code:: python

            if abs(min_delta_x) <= 1e-8:
                next_x_new = self.space.rvs(random_state=self.rng)[0]
                warnings.warn("The objective has been evaluated at point {} "
                              "before, using {} instead"
                              .format(next_x, next_x_new))
                next_x = next_x_new

    """
    import skopt
    import skopt.plots
    import matplotlib as mpl
    import matplotlib.pyplot as plt  # Plotting library
    import pickle

    def plot_save(filepath, dpi=200, transparent=False,
                  extensions=['.png', '.svg']):
        """Save plot figures to dífferent file formats."""
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        for ext in extensions:
            try:
                plt.savefig(filepath+ext, dpi=dpi, bbox_inches='tight',
                            transparent=transparent)
            except Exception as e:
                logger.error(e)

    mpl.rcParams['font.size'] = 5  # Matplotlib setup: For evaluation plots

    if n_cores == 0:  # Set number of CPU cores to use
        n_cores = multiprocessing.cpu_count() - 1

    if optimizer_pickle is None:
        # Start a fresh optimization
        sk_optimizer = skopt.Optimizer(
            dimensions=list(opt_dimensions.values()),
            random_state=random_state,
            **skopt_kwargs,
            )

        # Somewhat hacky, but here we store the name of each dimension
        # in the proper place, i.e. within the `space` object
        for i, name in enumerate(opt_dimensions.keys()):
            sk_optimizer.space.dimensions[i].name = name

    else:
        # Load existing optimizer results
        # result = skopt.load(optimizer_pickle)
        with open(optimizer_pickle, 'rb') as f:
            sk_optimizer = pickle.load(f)
        result = sk_optimizer.get_result()

        logger.info('Optimizer: Loaded existing optimizer results %s',
                    optimizer_pickle)

    # Start the optimization loop
    round_ = 1
    count = len(sk_optimizer.Xi)  # calls to evaluation function
    user_next_x = []  # Can be filled from yaml file
    user_ask = False  # user_next_x is not used by default
    eval_func_kwargs = dict()

    while count < n_calls:
        logger.info('Optimizer: Starting iteration round %s (%s of %s '
                    'simulations done)', round_, count, n_calls)

        if user_ask and len(user_next_x) > 0:
            next_x = user_next_x
            logger.info('Simulating this round with user input: %s', next_x)

        else:
            try:  # get points to evaluate
                next_x = sk_optimizer.ask(n_points=n_cores)

            except ValueError as ex:
                logger.exception(ex)
                raise
                # continue

        try:
            param_table = pd.DataFrame.from_records(
                    next_x, columns=sk_optimizer.space.dimension_names)
        except Exception as ex:
            logger.exception(ex)
            continue

        round_ += 1  # increment round counter
        count += len(next_x)  # counter for calls to evaluation function
        # evaluate points in parallel
        next_y = eval_func(param_table, **eval_func_kwargs)
        result = sk_optimizer.tell(next_x, next_y)
        result.nit = count

        if result.fun < tol:
            result.success = True
            break
        else:
            result.success = False

        # Save intermediate results after each round as pickle objects
        if plots_dir is not None and not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        if plots_dir is not None:
            # skopt.utils.dump(result,
            #                  os.path.join(plots_dir, 'optimizer.pkl'),
            #                  store_objective=True)
            with open(os.path.join(plots_dir, 'optimizer.pkl'), 'wb') as f:
                pickle.dump(sk_optimizer, f)

        # Generate and save optimization result plots:
        if result.space.n_dims > 1:
            try:
                plt.close('all')
            except Exception:
                pass

            plots_dir = os.path.abspath(plots_dir)
            if plots_dir is not None:
                try:
                    skopt.plots.plot_evaluations(result)
                    plot_save(os.path.join(plots_dir, 'skopt_evaluations'))
                except OSError as e:
                    logger.error(e)

                try:  # plot_objective fails before n_initial_points are done
                    skopt.plots.plot_objective(result)
                except IndexError:
                    logger.info(
                        'Not yet enough data to plot partial dependence.')
                    plt.figure()
                else:
                    try:
                        plot_save(os.path.join(plots_dir, 'skopt_objective'))
                    except OSError as e:
                        logger.error(e)

                try:
                    plt.close('all')
                    fig_conv = skopt.plots.plot_convergence(result)
                    fig_conv.set_yscale(yscale)
                    plot_save(os.path.join(plots_dir, 'skopt_convergence'))
                except Exception as e:
                    logger.error(e)

        # A yaml file in the current working directory allows to manipulate
        # the optimization during runtime:
        # Change n_cores and acquisition function arguments, set next points,
        # terminate optimizer
        try:
            opt_dict = yaml.load(open(opt_cfg, 'r'), Loader=yaml.FullLoader)

            # Overwrite number of cores with YAML setting
            n_cores_last = n_cores
            n_cores = opt_dict.setdefault('n_cores', n_cores)
            if n_cores_last != n_cores:
                logger.info('Change number of cores from %s to %s',
                            n_cores_last, n_cores)

            # Change kappa and xi on the go (exploration vs. exploitation)
            acq_func_last = sk_optimizer.acq_func
            sk_optimizer.acq_func = opt_dict.setdefault(
                'acq_func', sk_optimizer.acq_func)
            if acq_func_last != sk_optimizer.acq_func:
                logger.info('Change acquisition function from %s to %s',
                            acq_func_last, sk_optimizer.acq_func)

            acq_func_kwargs_last = sk_optimizer.acq_func_kwargs
            sk_optimizer.acq_func_kwargs = opt_dict.setdefault(
                'acq_func_kwargs', sk_optimizer.acq_func_kwargs)
            if acq_func_kwargs_last != sk_optimizer.acq_func_kwargs:
                logger.info('Change acquisition function arguments from %s to '
                            '%s',
                            acq_func_kwargs_last, sk_optimizer.acq_func_kwargs)
            sk_optimizer.update_next()

            # Next evaluation points given as user input
            user_ask = opt_dict.setdefault('user_ask', False)  # Boolean
            args_list = opt_dict.setdefault('user_range_prod', [])  # range
            try:
                # Input must be given in standard range() notation as a list
                # for each dimension
                user_next_x = convert_user_next_ranges(args_list)
            except Exception as ex:
                logger.exception(ex)

            # To further customize the execution of the evaluation function,
            # we can load any dict-styled keyword arguments from the YAML
            # to pass on to the eval_func:
            if user_ask:
                eval_func_kwargs = opt_dict.setdefault('eval_func_kwargs',
                                                       dict())
            else:
                eval_func_kwargs = dict()

            # Take note: Kill the optimizer
            kill = opt_dict.setdefault('kill', False)

            # Reset some values to default
            opt_dict['user_ask'] = False  # Reset the boolean
            opt_dict['kill'] = False

            # Save file with changed settings
            yaml.dump(opt_dict, open(opt_cfg, 'w'),
                      default_flow_style=None)

            if kill:
                logger.critical('Optimizer: Killed by file %s', opt_cfg)
                break
        except Exception:
            pass

    logger.info('Optimizer: Best fit after %s simulations: %s\n%s',
                count, result.fun,
                pd.Series(data=result.x,
                          index=result.space.dimension_names).to_string()
                )

    if result.space.n_dims > 1:
        if plots_show is True:
            # Show optimization result plots
            plt.show()

    return result


def convert_user_next_ranges(args_list):
    """Convert the ranges for next optimization as defined by the user.

    For each entry in the given list, use the items in that entry as input
    for the range() function (start, stop, step). Then get the product of
    all those ranges. This allows to manually define a grid of points for the
    optimizer to simulate.

    While this works, using ``skopt_optimize(initial_point_generator="grid"``)
    is much easier!
    """
    import itertools
    if len(args_list) > 0:
        # ranges = [range(*items) for items in args_list]
        ranges = []
        for items in args_list:
            if isinstance(items, list):
                ranges.append(range(*items))
            else:
                ranges.append([items])
        combis = list(itertools.product(*ranges))
        return combis
    else:
        return []


if __name__ == "__main__":
    """This is executed when the script is started directly with
    Python, not when it is loaded as a module.
    """
    # Define output format of logging function
    logging.basicConfig(format='%(asctime)-15s %(message)s')
