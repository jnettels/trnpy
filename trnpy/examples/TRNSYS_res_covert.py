# Copyright (C) 2020 Joris Nettelstroth

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

"""Find TRNSYS .out-files in the current folder and convert them to Excel.

This example is prepared for output files with two header rows, describing
name and unit of each column.

This script also allows creating interactive timeline plots in HTML by using
the library Bokeh.
"""

import os
import glob
import logging
import pandas as pd

from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool, RangeTool
from bokeh.layouts import layout, column
from bokeh.palettes import viridis
from bokeh.io import save

# Define the logging function
logger = logging.getLogger(__name__)


def main():
    """Run the main function."""
    files = get_result_files()
    df_list = read_TRNSYS_results(files)
    save_Excel_files(df_list, files)

    create_bokeh_htmls(df_list, files)


def get_result_files(search_filter='./*.out'):
    """Find all the files that match a certain filter.

    Args:
        search_filter (str, optional): Filter for file search. Defaults
        to './*.out'.

    Returns:
        files (list): List of found file paths.

    """
    files = glob.glob(search_filter)  # Find all files that match the filter
    if len(files) > 0:
        logger.info(files)  # Print confirmation to the console
    else:
        logger.error('No file found matching the search filter.')

    return files


def read_TRNSYS_results(files, drop_unit_row=False):
    """Convert all files in the given list to Excel.

    Args:
        files (list): List of file paths with files to load.

        drop_unit_row (bool, optional): Drop the second row of data, which
        contains units. Defaults to False.

    Returns:
        df_list (list): List of Pandas DataFrames.

    """
    df_list = []  # Prepare a list of DataFrames
    for file in files:  # Read each file in the list of files
        df = pd.read_csv(file,  # Use Pandas to read the csv file
                         sep=r'\t*\s+',  # Seperator is a mix of tabs and space
                         engine='python',
                         header=[0, 1],  # First two rows are headers
                         index_col=0,  # Use TIME column as index
                         )
        if drop_unit_row:
            df.columns = df.columns.droplevel(1)  # Drop the row with units

        df.columns.set_names(['name', 'unit'], inplace=True)
        df.index.set_names('TIME', inplace=True)  # Set the name of the index

        df_list.append(df)  # Add the current DataFrame to the list

        if logger.isEnabledFor(logging.DEBUG):  # Only do if in DEBUG mode
            logger.debug(file)
            print(df)  # Print the DataFrame to the console

    return df_list


def save_Excel_files(df_list, files, subfolder='Excel'):
    """Save the given DataFrames to a subfolder.

    Args:
        df_list (list): List of Pandas DataFrames.

        files (list): List of original filenames.

        subfolder (str, optional): Name of subfolder to use for output.
        Defaults to 'Excel'.

    Returns:
        None.

    """
    for df, file in zip(df_list, files):
        # Construct the output path
        filename = os.path.splitext(os.path.basename(file))[0] + '.xlsx'
        filepath = os.path.join(os.path.dirname(file), subfolder, filename)

        # Create the output folder, if it does not already exist
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        # Finally, save the DataFrame to an Excel file
        logger.info('Saving {}'.format(filepath))
        df.to_excel(filepath)


def create_bokeh_htmls(df_list, files, subfolder='Bokeh', html_show=False):
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
        logger.info('Saving {}'.format(filepath))
        create_bokeh_html(df, title=os.path.basename(file),
                          html_filename=filepath, html_show=html_show)


def create_bokeh_html(df, title='Bokeh',
                      html_filename='bokeh.html',
                      html_show=True, sync_xaxis=True):
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

    Returns:
        None.

    """
    fig_list = create_bokeh_timelines(df, sync_xaxis=sync_xaxis)

    # Define the layout with all elements
    doc_layout = layout(fig_list, sizing_mode='stretch_both')
    # Create the output file
    output_file(html_filename, title=title)
    if html_show:
        show(doc_layout)  # Trigger opening a browser window with the html
    else:
        save(doc_layout)  # Only save the html without showing it


def create_bokeh_timelines(df, sync_xaxis=True):
    """Create Bokeh plots from a Pandas DataFrame.

    Args:
        df (DataFrame): Pandas DataFrame with timeseries data to be plotted.

        sync_xaxis (bool, optional): Synchronize the x-axis of the plots.
        This is a nice feature, but impacts performance. Defaults to False.

    Returns:
        None.

    """
    # Determine the type of the x axis
    if df.index.dtype == 'datetime64[ns]':
        x_axis_type = 'datetime'
    else:
        x_axis_type = 'linear'

    # For DataFrame with Multiindex, get a list of unique units
    unit_list = [unit for unit in df.columns.get_level_values('unit').unique()]

    # For each DataFrame, create a bokeh figure
    fig_list = []
    for unit in unit_list:
        # Make a cross-selection with the current unit
        df_unit = df.xs(unit, level='unit', axis='columns')
        if len(fig_list) > 0:
            # Synchronize the x_ranges of all subsequent figures to the first
            if sync_xaxis:
                fig_link = fig_list[0].children[0]
        else:
            fig_link = None
        p = create_bokeh_timeline(df_unit, fig_link=fig_link, y_label=unit,
                                  x_axis_type=x_axis_type)
        fig_list.append(p)

    return fig_list


def create_bokeh_timeline(df, fig_link=None, y_label=None, title=None,
                          output_backend='canvas', x_axis_type='linear',
                          **kwargs):
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
        None.

    """
    x_col = df.index.name  # Get name of index (used as x-axis)
    if x_col is None:
        logger.error('No index name found, plot will not show properly!')
    source = ColumnDataSource(df)  # Bokeh's internal data structure

    if fig_link is None:  # Set a default range for the x axis
        fig_x_range = (df.index.min(), df.index.max())
    else:  # link to the range of the given figure
        fig_x_range = fig_link.x_range

    # Create the primary plot figure
    p = figure(plot_width=1000, plot_height=250, x_axis_type=x_axis_type,
               x_range=fig_x_range, output_backend=output_backend,
               title=title, **kwargs)

    # Create the glyphs in the figure (one line plot for each data column)
    y_cols = list(df.columns)
    palette = viridis(len(y_cols))  # Generate color palette of correct length
    for y_col, color in zip(y_cols, palette):
        r = p.line(x_col, y_col, source=source, color=color, name=y_col,
                   legend_label=y_col)
        # Add a hover tool to the figure
        add_hover_tool(p, renderers=[r], x_col=x_col, x_axis_type=x_axis_type)

    custom_bokeh_settings(p)  # Set additional features of the plot

    if y_label is not None:
        p.yaxis.axis_label = y_label

    select = get_select_RangeTool(p, x_col, y_cols, source, palette,
                                  output_backend, x_axis_type=x_axis_type)

    return column([p, select], sizing_mode='stretch_both')


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
        select (TYPE): DESCRIPTION.

    """
    select = figure(plot_height=45, y_range=p.y_range, tools="",
                    x_axis_type=x_axis_type, y_axis_type=None,
                    toolbar_location=None, background_fill_color="#efefef",
                    output_backend=output_backend,
                    height_policy="fixed", width_policy="fit",
                    )
    for y_col in y_cols:  # Show all lines of primary figure in "select", too
        select.line(x_col, y_col, source=source)

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
    # p.yaxis.major_label_orientation = "vertical"
    # p.xaxis.major_label_orientation = 1.2
    # p.legend.background_fill_alpha = 0.5
    # p.legend.location = "top_left"
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"  # clickable legend items
    p.legend.spacing = 0
    p.legend.padding = 2
    # p.legend.label_text_font_size = '8pt'
    # p.toolbar.logo = None  # Remove Bokeh logo


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


def setup():
    """Set up the logger."""
    # Define the logging function
    logging.basicConfig(format='%(asctime)-15s %(levelname)-8s %(message)s')
    log_level = 'DEBUG'
    # log_level = 'INFO'
    logger.setLevel(level=log_level.upper())  # Logger for this module


if __name__ == '__main__':
    setup()
    main()
