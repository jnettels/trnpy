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

Module Polyfit
-----------
This is a collection of functions for working with the results of
optimizations in TRNpy. Most importantly, ``poly_fit_and_plot()`` allows
to create fitted polynoms of from data in 2 or 3-dimensional space.

These functions are quite experimental and adapted to specific tasks.

'''
import numpy as np
import pandas as pd
import os
import matplotlib as mpl         # Matplotlib
import matplotlib.pyplot as plt  # Plotting library
import logging

# Define the logging function
logger = logging.getLogger(__name__)


def make_bokeh_colorbar(df, x, y, z, file=None):
    '''Bokeh plot with colorbar.
    '''
    from bokeh.models import (ColumnDataSource, LinearColorMapper, BasicTicker,
                              ColorBar, HoverTool)
    from bokeh.plotting import figure, output_file, show

    source = ColumnDataSource(data=df)

    color_mapper = LinearColorMapper(palette='Plasma256',
                                     low=df[z].min(),
                                     high=df[z].max())
    p = figure()
    p.circle(x=x, y=y, source=source, size=20, line_color=None,
             fill_color={'field': z, 'transform': color_mapper})

    p.xaxis.axis_label = x
    p.yaxis.axis_label = y

    color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                         label_standoff=12, border_line_color=None,
                         location=(0, 0), title=z)

    p.add_layout(color_bar, 'left')

    # Add the hover tool
    hover = HoverTool(point_policy='snap_to_data',
                      tooltips=[(x, '@'+x),
                                (y, '@'+y),
                                (z, '@'+z)]
                      )
    p.add_tools(hover)

    if file is not None:
        output_file(os.path.join(os.path.dirname(file), 'color_bar.html'))

    show(p)


def polyfit2d(x, y, f, order):
    '''Fit given data to a 2-D polynomial of given order. Return array of
    coefficients ``c`` as expected as input by numpy's polyval2d.

    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * x^i * y^j

    Source:
    https://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent

    Numpy polyval2d:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.polynomial.polynomial.polyval2d.html
    '''
    import numpy as np

    x = np.asarray(x)
    y = np.asarray(y)
    f = np.asarray(f)
    deg = np.array([order, order])
    vander = np.polynomial.polynomial.polyvander2d(x, y, deg)
    vander = vander.reshape((-1, vander.shape[-1]))
    f = f.reshape((vander.shape[0],))
    c = np.linalg.lstsq(vander, f, rcond=None)[0]
    return c.reshape(deg+1)


def poly_fit_and_plot(x, y, z=None, **kwargs):

    if z is None:
        func = poly1d_fit_and_plot(x=x, y=y, **kwargs)

    else:
        func = poly2d_fit_and_plot(x=x, y=y, z=z, **kwargs)

    return func


def format_axis_futureSuN(ax):
    '''futureSuN style format:

        * space as thousands separator
        * rotation
    '''
    import locale
    locale.setlocale(locale.LC_ALL, '')  # Use space as thousands separator
    ax.get_xaxis().set_major_formatter(  # Use space as thousands separator
        mpl.ticker.FuncFormatter(lambda x, p: format(int(x), 'n')))
    ax.get_yaxis().set_major_formatter(  # Use space as thousands separator
        mpl.ticker.FuncFormatter(lambda x, p: format(int(x), 'n')))
    ax.xaxis.set_tick_params(rotation=30)  # rotation is useful sometimes


def poly1d_fit_and_plot(
        x, y, c_axis=None, order=2,
        x_label='', y_label='', z_label='',
        title=None, label_fit='Fit', nx=20, ny=None, show_plot=True,
        print_formula=False, input_fig_ax=None,
        limits_xy=None, plot_fit_curve=False,
        plot_scatter_solution=False,
        plot_scatter=True, savedir=False, savefile='2D-plot.png',
        savefig_args=dict(dpi=500, bbox_inches='tight', pad_inches=0.05),
        contour_lines=False, color_scatter='red',
        ):
    r'''Fit given data to a 1-D polynomial of given order. Plot input data
    and fittet data. Return a function representing the following equation:

    .. math:: p(x) = \sum_{i} c_{i} \cdot x^i

    For order < 0, fit instead to a different type of power function:

    .. math:: p(x) = a \cdot x^b + c

    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    try:
        if len(x) == 0:
            return None
    except TypeError:
        if isinstance(x, float):
            x = pd.Series(x)
        if isinstance(y, float):
            y = pd.Series(y)

    # Create 2d plot of input data
    if input_fig_ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.gca()
    else:
        # Plot all following elements onto an existing figure
        fig, ax = input_fig_ax

    if plot_scatter:
        if c_axis is not None:
            paths = ax.scatter(x, y, label=str(len(x))+' Lösungen',
                               c=c_axis, cmap='plasma')
#            paths = ax.scatter(x[1:], y[1:], label=str(len(x)-1)+' Lösungen',
#                               c=c_axis[1:], cmap='plasma')
            fig.colorbar(  # regular colorbar
                    paths,
                    ax=ax,
                    # fraction=0.046, pad=0.04,
                    label=z_label)

            if contour_lines:
                CS = ax.tricontour(x, y, c_axis, cmap='plasma')
                plt.clabel(CS, inline=1, fontsize=10)

        else:
            ax.scatter(x, y, label=str(len(x))+' Lösungen', c=color_scatter)

    if plot_scatter_solution:
        ax.scatter(x[:1], y[:1], c='green', label='Optimum', s=100, zorder=10)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    if limits_xy is not None:
        ax.axis(limits_xy)  # tuple of xmin, xmax, ymin, ymax
    format_axis_futureSuN(ax)

    # Perform fitting
    if order >= 0:
        c = np.polyfit(x, y, deg=order)
        poly = np.poly1d(c)
        if print_formula:
            print('x = '+x_label+'; y = '+y_label)
            print(np.poly1d(poly))

    elif order == -3:
        def func_fit(x, a, b, c):
            return a * np.power(x, b) + c
        popt, pcov = curve_fit(func_fit, x, y, maxfev=9000000, method='trf')

        def poly(x):
            return func_fit(x, *popt)
        if print_formula:
            print('x = '+x_label+'; f(x) = '+y_label)
            print('a * x^b + c:', str(popt))

    elif order == -4:
        def func_fit(x, a, b, c, d):
            return a * np.power(x+b, c) + d
        popt, pcov = curve_fit(func_fit, x, y, maxfev=9000000, method='trf')

        def poly(x):
            return func_fit(x, *popt)
        if print_formula:
            print('x = '+x_label+'; f(x) = '+y_label)
            print('a * (x+b)^c + d:', str(popt))

    elif order == -5:
        def func_fit(x, a, b, c, d, e):
            return a * np.power(x+b, c) + d + e*x
        popt, pcov = curve_fit(func_fit, x, y, maxfev=9000000, method='trf')

        def poly(x):
            return func_fit(x, *popt)
        if print_formula:
            print('x = '+x_label+'; f(x) = '+y_label)
            print('a * (x+b)^c + d + e*x:', str(popt))

    else:
        raise ValueError('Order for fitting not defined: '+str(order))

    # Continue plotting
    xx = np.linspace(x.min(), x.max(), nx)
    yy = poly(xx)
    next(ax._get_lines.prop_cycler)['color']
    if plot_fit_curve:
        ax.plot(xx, yy, label=label_fit)

    plt.legend(loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.1))
    if show_plot:
        plt.show()
    if savedir:
        fig.savefig(os.path.join(savedir, savefile),
                    **savefig_args)

    def func(xi, check_confidence=True, filter_confidence=False,
             show_plot=False):
        '''Return the results of the fitted polynom, calculated for the given
        input.
        '''
        result = poly(xi)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.gca()
        xx = np.linspace(x.min(), x.max(), nx)
        yy = poly(xx)
        ax.plot(xx, yy, color='#ff8021', label='Fit')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        format_axis_futureSuN(ax)

        inside = xi[(xi >= x.min()) & (xi <= x.max())]
        outside = xi[(xi < x.min()) | (xi > x.max())]

        if check_confidence:
            ax.scatter(inside, result[inside.index], color='green',
                       label='Auswahl')
            if len(outside) > 0:
                ax.scatter(outside, result[outside.index], color='red',
                           label='Auswahl')
        else:
            ax.scatter(xi, result[xi.index], color='green', label='Auswahl')

        plt.legend(loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.1))

        if show_plot:
            plt.show()
        if savedir:
            fig.savefig(os.path.join(savedir, 'fitted_data.png'),
                        **savefig_args)

        if check_confidence:
            result[outside.index] = float('NaN')

        return result

    return func


def poly2d_fit_and_plot(x, y, z, order=2, x_label='', y_label='', z_label='',
                        title='', nx=20, ny=20, show_plot=True, c_axis=None,
                        print_formula=False, plot_fit_curve=True,
                        savedir=False, savefile='3D-plot.png',
                        savefig_args=dict()):
    '''Fit given data to a 2-D polynomial of given order. Plot input data
    and fittet data. Return a function representing the following equation:

    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * x^i * y^j
    '''
    import matplotlib.path as mplPath
    from mpl_toolkits.mplot3d import Axes3D  # for plot_trisurf
    from matplotlib import cm

    def make_colormap(seq):
        """Return a LinearSegmentedColormap
        seq: a sequence of floats and RGB-tuples. The floats should be
        increasing and in the interval (0,1).
        """
        import matplotlib.colors as mcolors
        cdict = {'red': [], 'green': [], 'blue': []}

        # make a lin_space with the number of records from seq.
        x = np.linspace(0, 1, len(seq))

        for i in range(len(seq)):
            segment = x[i]
            tone = seq[i]
            cdict['red'].append([segment, tone, tone*.5])
            cdict['green'].append([segment, tone, tone])
            cdict['blue'].append([segment, tone, tone])

        return mcolors.LinearSegmentedColormap('CustomMap', cdict)

    if c_axis is None:
        color_map = cm.plasma
    else:
        color_map = make_colormap(c_axis/max(c_axis))

    # Create 3d plot of input data
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(x, y, z, cmap=color_map,
                           linewidth=0, antialiased=True)

    if c_axis is not None:
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Add annotations
    ax.text2D(0.01, 0.99, title, transform=ax.transAxes)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    if order >= 0 and plot_fit_curve:
        # Fit a 2d polynomial of given order
        c = polyfit2d(x, y, z, order=order)  # get array of coefficients

        # Evaluate it on a grid...
        xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx),
                             np.linspace(y.min(), y.max(), ny))
        zz = np.polynomial.polynomial.polyval2d(xx, yy, c)

        # Add 3d plot of fitted data
        x_list = [item for sublist in xx for item in sublist]
        y_list = [item for sublist in yy for item in sublist]
        z_list = [item for sublist in zz for item in sublist]
        ax.set_zlim([z.min()/1.1, z.max()*1.1])
        surf = ax.plot_trisurf(x_list, y_list, z_list, cmap=cm.viridis,
                               linewidth=0, antialiased=True, alpha=0.8)

        # Alternative: 2-D contour plot
        plt.figure()
        CS = plt.contour(xx, yy, zz)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(z_label)

    else:
        # order < 0: Do not perform fit
        c = np.array([[0]])

    if show_plot:
        plt.show()
    if savedir:
        # Set rotation angle to 30 degrees
        ax.view_init(azim=230)
        fig.savefig(os.path.join(savedir, savefile), **savefig_args)

    if print_formula:
        # Create string representation of the full formula
        forumula1 = 'f(x,y) = 0'
        forumula2 = 'f(x,y) = 0'
        for i, row in enumerate(c):
            for j, val in enumerate(row):
                forumula1 += ' + '+str(val)+'*x^'+str(i)+'*y^'+str(j)
                forumula2 += ' + '+str(val)+'*pow(x, '+str(i)+')*pow(y, '\
                             + str(j)+')'

        txt = 'x = '+x_label+'; y = '+y_label+'; f(x,y) = '+z_label+'\n'\
              + forumula1 + '\n' + forumula2 + '\n'
        logger.info('Formula for 2D polynomial in '+savefile+':\n'+txt)
        if savedir:
            with open(os.path.join(savedir, 'formulae.txt'), 'a+') as f:
                f.write(savefile+':\n')
                f.write(txt+'\n')

    def func(xi, yi, check_confidence=False, show_plot=False,
             contains_radius=1, savefile=False, savefig_args=dict()):
        '''Return the results of the fitted polynom, calculated for the given
        input.

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html

        TODO: The hull vertices are not rendered as a closed line
        '''
        import matplotlib.patches as patches
        from scipy.spatial import ConvexHull

        result = np.polynomial.polynomial.polyval2d(xi, yi, c)

        if check_confidence:
            # Create coloured points that marks the confidence area
            points = np.array([x, y]).T
            hull = ConvexHull(points)
            # vertices are a list of coordinates that mark the boundaries
            vertices = np.array([points[hull.vertices, 0],
                                 points[hull.vertices, 1]]).T
            bbPath = mplPath.Path(vertices)

#            plt.figure()
            zlim_low = ax.get_zlim()[0]  # Show boundaries at bottom of z-axis
            ax.plot(points[hull.vertices, 0], points[hull.vertices, 1],
                    zlim_low, 'r--', lw=2)  # Plot boundaries as red line (3D)
            fig2, ax2 = plt.subplots()
            ax2.plot(points[hull.vertices, 0], points[hull.vertices, 1],
                     'r--', lw=2)  # Plot boundaries as red line (2D)
            patch = patches.PathPatch(bbPath, facecolor='orange', alpha=0.1,
                                      lw=1)
            ax2.add_patch(patch)
            ax2.set_xlim(x.min(), x.max())
            ax2.set_ylim(y.min(), y.max())
            # Check if points xi, yi are contained within bbPath
            check = bbPath.contains_points(np.array([xi, yi]).T,
                                           radius=contains_radius,
                                           )
            for i, point in enumerate(np.array([xi, yi]).T):
                if check[i]:
                    ax.scatter(point[0], point[1], zlim_low,
                               s=40, marker='^', color='green')
                    ax2.scatter(point[0], point[1], s=40, marker='^',
                                color='green')
                else:
                    ax.scatter(point[0], point[1],  zlim_low,
                               s=40, marker='^', color='red')
                    ax2.scatter(point[0], point[1], s=40, marker='^',
                                color='red')
            if show_plot:
                plt.show()

            if savefile:
                fig.savefig(savefile, **savefig_args)

            for i, confidence in enumerate(check):
                if confidence == False:  # 'confidence is False' fails
                    result[i] = float('NaN')

        return result

    return func  # Return the function object


def make_nomograph(func, file):
    '''Create nomograph with pynomo.

    Variant: x-axis: V_Sp, y-axis: A_PV, contour: WP_multi

    - Coordinate x is referenced with ``wd`` in block parameters
    - Coordinate y is referenced with ``u`` in block parameters
    - Coordinate z is referenced with ``v`` in block parameters

    - v_func is the function f(x,y) which returns z
    - v_func is the function f(wd,v) which returns u
    '''
    import sys
    sys.path.insert(0, "..")
    from pynomo.nomographer import Nomographer

    def func2(x, y):
        return func([x], [y])[0]

    isopleth_wd = 30000  # x
    isopleth_v = 56  # z
    print(func2(isopleth_wd, isopleth_v))

    block_params = {
       'block_type': 'type_5',
       'v_func': func2,
       'u_values': [200, 250, 300, 350, 400, 450, 500],  # y-Axis
       'v_values': [48, 56, 64, 72],  # , 80, 90],  # z-Axis
       'wd_tick_levels': 2,
       'wd_tick_text_levels': 1,
       'wd_tick_side': 'right',
       'wd_title': r'$V_{Sp}$',  # x-Axis
       'v_title': r'$WP_{m}$',  # y-Axis
       'u_title': r'$A_{PV}$',  # z-Axis
       'u_title_distance_center': 2.0,
       'v_title_distance_center': 2.0,
       'v_title_opposite_tick': True,
       'wd_title_opposite_tick': True,
       'wd_title_distance_center': 3.0,
       'isopleth_values': [['unknown_u', isopleth_v, isopleth_wd]],
     }

    main_params = {
        'filename': os.path.join(os.path.dirname(file), 'polyfit.pdf'),
        'paper_height': 10.0,
        'paper_width': 10.0,
        'block_params': [block_params],
        'transformations': [('rotate', 0.01), ('scale paper', )]
        }

    Nomographer(main_params)


def plot_contour(x, y, z, x_label='', y_label='', z_label='', limits_xy=None,
                 plot_empty=False, contour_tricontour=False,
                 contour_lines=False, ):
    '''Create matplotlib contour plots along with a colorbar next to the plot.
    '''

    fig = plt.figure()
#    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    if limits_xy is not None:  # Set axis limits
        ax.axis(limits_xy)  # tuple of xmin, xmax, ymin, ymax
    format_axis_futureSuN(ax)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

#    n_levels = np.linspace(0, 0.48, num=49)  # For colorbar
    n_levels = np.linspace(0, 0.6, num=55)  # For colorbar
#    n_levels = np.linspace(0, 0.6, num=11)  # For colorbar

    if plot_empty:
        ax.plot(x, y, 'ko', ms=0, label='Simulationen')  # invisible dots
#        ax.set_title('%d Simulationen' % len(x))

        # Manualy draw colorbar on empty plot
        cax, _ = mpl.colorbar.make_axes(ax)
        norm = mpl.colors.Normalize(vmin=min(n_levels), vmax=max(n_levels))
        mpl.colorbar.ColorbarBase(cax, cmap="viridis_r", norm=norm,
                                  label='Fehler',
                                  format='%.2f',
                                  ticks=np.linspace(0, 0.6, num=10),
                                  )
    else:
        ax.plot(x, y, 'ko', ms=3, label='%d Simulationen' % len(x))

        if contour_lines:
            CS = ax.tricontour(x, y, z, levels=n_levels, linewidths=0.5,
                               colors='k')
            plt.clabel(CS, inline=1, fontsize=10)

        cntr = ax.tricontourf(x, y, z, levels=n_levels, cmap="viridis_r")
        fig.colorbar(cntr, label='Fehler',
                     ax=ax,
                     format='%.2f',
#                     fraction=0.046, pad=0.04,
#                     cax = fig.add_axes([1, 0, 0.1, 1])  # TODO
                     )  # regular colorbar

    plt.legend(loc='upper center', ncol=8, bbox_to_anchor=(0.5, 1.1))

    return fig, ax


def build_gif(df, x, y, savedir, x_label='', y_label='',
              speed_double=False, speed_test=False, limits_xy=None,
              savefig_args=dict(dpi=100, bbox_inches='tight', pad_inches=0.05,
                                format='png'),
              ):
    '''Create GIF animations from a DataFrame containing the "history" of an
    optimization run
    '''
    import imageio
    import io
    import shutil

    if limits_xy is None:
        limits_xy = (min(df[x]), max(df[x]), min(df[y]), max(df[y]))

    with imageio.get_writer(os.path.join(savedir, '1_contour.gif'),
                            mode='I') as writer:
        for i in range(len(df)):
            if logger.isEnabledFor(logging.INFO):
                print(' GIF', i, len(df), end='\r')
            if i < 3:
                continue
            if speed_test:
                if i > 15:
                    break
            if speed_double:
                if not i % 2:
                    continue
            fig, ax = plot_contour(df.loc[:i, x], df.loc[:i, y],
                                   df.loc[:i, 'error'], x_label=x_label,
                                   y_label=y_label, z_label='Fehler',
                                   limits_xy=limits_xy)
#            plt.show()
            buffer = io.BytesIO()
            fig.savefig(buffer, **savefig_args)
            plt.clf()
            plt.close('all')
            buffer.seek(0)
            image = imageio.imread(buffer)
            writer.append_data(image)

        logger.info('GIF finished')

        filename = os.path.join(savedir, '2_contour_gif_end.png')
        with open(filename, 'wb') as f:
            buffer.seek(0)
            shutil.copyfileobj(buffer, f, length=131072)
            buffer.close()

    # Also create an empty starting plot for the GIF
    fig, ax = plot_contour(df[x], df[y], df['error'], x_label=x_label,
                           y_label=y_label, z_label='Fehler',
                           limits_xy=limits_xy, plot_empty=True)
    filename = os.path.join(savedir, '0_contour_gif_start.png')
    fig.savefig(filename, **savefig_args)
