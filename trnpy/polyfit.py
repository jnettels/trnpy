# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 14:17:44 2018

@author: nettelstroth

Library of functions for working with the results of optimizations in TRNpy.

"""
import numpy as np
import pandas as pd
import os
import trnpy
import pickle
import matplotlib as mpl         # Matplotlib
import matplotlib.pyplot as plt  # Plotting library


def make_bokeh_colorbar(df, x, y, z):
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
    c = np.linalg.lstsq(vander, f)[0]
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
    ax.get_xaxis().set_major_formatter(  # Use space as thousands separator
        mpl.ticker.FuncFormatter(lambda x, p: format(int(x), 'n')))
    ax.get_yaxis().set_major_formatter(  # Use space as thousands separator
        mpl.ticker.FuncFormatter(lambda x, p: format(int(x), 'n')))
    ax.xaxis.set_tick_params(rotation=30)  # rotation is useful sometimes


def poly1d_fit_and_plot(x, y, order=2, x_label='', y_label='', z_label='',
                        title=None, nx=20, ny=20, show_plot=True,
                        print_formula=False, input_fig_ax=None,
                        limits_xy=None):
    r'''Fit given data to a 1-D polynomial of given order. Plot input data
    and fittet data. Return a function representing the following equation:

    .. math:: p(x) = \sum_{i} c_{i} \cdot x^i

    For order < 0, fit instead to a different type of power function:

    .. math:: p(x) = a \cdot x^b + c

    '''
    import numpy as np
    import matplotlib.pyplot as plt
    import locale
    from scipy.optimize import curve_fit
    locale.setlocale(locale.LC_ALL, '')  # Use space as thousands separator

    # Create 2d plot of input data

    if input_fig_ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.gca()
    else:
        # Plot all following elements onto an existing figure
        fig, ax = input_fig_ax

#    ax.scatter(x, y, label='Lösungen')
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

    else:
        def func_fit(x, a, b, c):
            return a * np.power(x, b) + c
        popt, pcov = curve_fit(func_fit, x, y, maxfev=9000000, method='trf')

        def poly(x):
            return func_fit(x, *popt)

    # Continue plotting
    xx = np.linspace(x.min(), x.max(), nx)
    yy = poly(xx)
    next(ax._get_lines.prop_cycler)['color']
#    ax.plot(xx, yy, label='Fit')

    if show_plot:
        plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.1))
        plt.show()
        fig.savefig(os.path.join(os.path.dirname(file), '2D-plot.png'),
                    dpi=500, bbox_inches='tight', pad_inches=0.05)

    if print_formula:
        if order >= 0:
            print(np.poly1d(poly))
        else:
            print('a * x^b + c')
            print(popt)

    def func(xi, check_confidence=True):
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

        if check_confidence:
            inside = xi[(xi >= x.min()) & (xi <= x.max())]
            outside = xi[(xi < x.min()) | (xi > x.max())]

            ax.scatter(inside, result[inside.index], color='green',
                       label='Auswahl')
            if len(outside) > 0:
                ax.scatter(outside, result[outside.index], color='red',
                           label='Auswahl')
        else:
            ax.scatter(xi, result[xi.index], color='green', label='Auswahl')

        plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.1))
        plt.show()
        fig.savefig(os.path.join(os.path.dirname(file), 'fitted_data.png'),
                    dpi=500, bbox_inches='tight', pad_inches=0.05)

        return result

    return func


def poly2d_fit_and_plot(x, y, z, order=2, x_label='', y_label='', z_label='',
                        title='', nx=20, ny=20, show_plot=True,
                        print_formula=False):
    '''Fit given data to a 2-D polynomial of given order. Plot input data
    and fittet data. Return a function representing the following equation:

    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * x^i * y^j
    '''
    import matplotlib.path as mplPath
    from mpl_toolkits.mplot3d import Axes3D  # for plot_trisurf
    from matplotlib import cm

    # Create 3d plot of input data
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, z, cmap=cm.plasma, linewidth=0, antialiased=True)

    # Add annotations
    ax.text2D(0.01, 0.99, title, transform=ax.transAxes)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    if order >= 0:
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
        ax.plot_trisurf(x_list, y_list, z_list, cmap=cm.viridis, linewidth=0,
                        antialiased=True)

        # Alternative: 2-D contour plot
        plt.figure()
        CS = plt.contour(xx, yy, zz)
    #    CS = plt.contour(xx, zz, yy)  # TODO debugging
        plt.clabel(CS, inline=1, fontsize=10)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(z_label)

    else:
        # order < 0: Do not perform fit
        c = np.array([[0]])

    if show_plot:
        plt.show()

    if print_formula:
        # Create string representation of the full formula
        forumula = 'f(x,y) = 0'
        for i, row in enumerate(c):
            for j, val in enumerate(row):
#                forumula += ' + '+str(val)+'*pow(x, '+str(i)+')*pow(y, '+str(j)+')'
                forumula += ' + '+str(val)+'*x^'+str(i)+'*y^'+str(j)
        print(forumula)

    def func(xi, yi, check_confidence=False):
        '''Return the results of the fitted polynom, calculated for the given
        input.

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
        '''
        import matplotlib.patches as patches
        from scipy.spatial import ConvexHull

        result = np.polynomial.polynomial.polyval2d(xi, yi, c)

        if check_confidence:
            points = np.array([x, y]).T
            hull = ConvexHull(points)
            vertices = np.array([points[hull.vertices, 0],
                                 points[hull.vertices, 1]]).T
            bbPath = mplPath.Path(vertices)

#            plt.figure()
            ax.plot(points[hull.vertices,0], points[hull.vertices,1], 0, 'r--', lw=2)
            fig2, ax2 = plt.subplots()
            ax2.plot(points[hull.vertices,0], points[hull.vertices,1], 0, 'r--', lw=2)
            patch = patches.PathPatch(bbPath, facecolor='orange', alpha=0.1, lw=1)
            ax2.add_patch(patch)
            ax2.set_xlim(x.min(), x.max())
            ax2.set_ylim(y.min(), y.max())
            check = bbPath.contains_points(np.array([xi, yi]).T)  # radius=100
            for i, point in enumerate(np.array([xi, yi]).T):
                if check[i]:
                    ax.scatter(point[0], point[1],  0, s=40, marker='^', color='green')
                    ax2.scatter(point[0], point[1], s=40, marker='^', color='green')
                else:
                    ax.scatter(point[0], point[1],  0, s=40, marker='^', color='red')
                    ax2.scatter(point[0], point[1], s=40, marker='^', color='red')
            plt.show()

            for i, confidence in enumerate(check):
                if confidence == False:  # 'confidence is False' fails
                    result[i] = float('NaN')

        return result

    return func  # Return the function object


def make_nomograph(func):
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


def plot_contour(x, y, z, x_label='', y_label='', z_label='', limits_xy=None):

    fig = plt.figure()
#    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    if limits_xy is not None:  # Set axis limits
        ax.axis(limits_xy)  # tuple of xmin, xmax, ymin, ymax
    format_axis_futureSuN(ax)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.plot(x, y, 'ko', ms=3, label='%d Simulationen' % len(x))
#    ax.plot(x, y, 'ko', ms=0, label='Simulationen')  # invisible dots

    n_levels = np.linspace(0, 0.48, num=49)
#    ax.tricontour(x, y, z, levels=n_levels, linewidths=0.5, colors='k')
    cntr = ax.tricontourf(x, y, z, levels=n_levels, cmap="viridis_r")
    fig.colorbar(cntr, ax=ax, label='Fehler')  # regular colorbar
    plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.1))

    # Manualy draw colorbar on empty plot
#    cax, _ = mpl.colorbar.make_axes(ax)
#    norm = mpl.colors.Normalize(vmin=min(n_levels), vmax=max(n_levels))
#    mpl.colorbar.ColorbarBase(cax, cmap="viridis_r", norm=norm, label='Fehler',
#                              ticks=np.linspace(0, 0.5, num=11),)

#    ax.set_title('%d Simulationen' % len(x))

    return fig, ax


def build_gif(df, x, y, x_label='', y_label=''):
    import imageio
    import io
    import shutil

    with imageio.get_writer(os.path.join(os.path.dirname(file), 'contour.gif'),
                            mode='I') as writer:
        for i in range(len(df)):
            print(' GIF', i, len(df), end='\r')
            if i < 3:
                continue
#            if i > 15:
#                break
#            if not i % 2:
#                continue
            fig, ax = plot_contour(df.loc[:i, x], df.loc[:i, y],
                                   df.loc[:i, 'error'], x_label=x_label,
                                   y_label=y_label, z_label='Fehler',
                                   limits_xy=(min(df[x]), max(df[x]),
                                              min(df[y]), max(df[y])))
#            plt.show()
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight',
                        pad_inches=0.05)
            plt.clf()
            plt.close('all')
            buffer.seek(0)
            image = imageio.imread(buffer)
            writer.append_data(image)

        filename = os.path.join(os.path.dirname(file), 'contour.png')
        with open(filename, 'wb') as f:
            buffer.seek(0)
            shutil.copyfileobj(buffer, f, length=131072)
            buffer.close()


if __name__ == "__main__":
    '''This is executed when the script is started directly with
    Python, not when it is loaded as a module.
    '''
    mpl.style.use('../SuN-Plants/futureSuN.mplstyle')  # Personalized matplotlib style file

# =============================================================================
#     file = os.path.join(
# #        r'C:\Trnsys17\Work\futureSuN\AP4\P2H_Quartier',
#         r'C:\Trnsys17\Work\futureSuN\AP4\P2H_Quartier\Result',
# #        r'C:\Trnsys17\Work\futureSuN\AP4\P2H_Quartier\Optimization\Result_181021 opt 0.9 CO2 ohne Einspeisung',
# #        r'Optimization\Result_opt_13 PV_WP',
# #        r'Optimization\Result_opt_14 VSp_WP',
# #        r'C:\Trnsys17\Work\futureSuN\AP4\Sonnenkamp\Result',
# #        'Result_180827 opt f_Sp=0.2',
# #        'Result_180827 opt f_Sp=0.2 V_Sp fix',
# #        'Result_opt_07',
# #        'Result_opt_08 (490)',
# #        'Result_opt_09',
# #        'Result_opt_10',
# #        'Result_opt_11 (644)',
#         'opt_res.xlsx'
# #        'opt_res_combine.xlsx'
#                         )
#     df = pd.read_excel(file)
# =============================================================================

    file = r'C:\Trnsys17\Work\futureSuN\AP4\P2H_Quartier\Result\opt_result.pkl'
    with open(file, 'rb') as f:
        opt_res = pickle.load(f)
    df = pd.DataFrame(opt_res.x_iters, columns=opt_res.labels)
    df['error'] = opt_res.func_vals
#    df.sort_values(by='error', inplace=True)


    # Old names
    #x = 'VSp_m3'
    #y = 'PV_A_brutto'
    #z = 'WP_multiplier_max'
#    x = 'PV_A_brutto'
#    y = 'WP_multiplier_max'
#    z = 'VSp_m3'
#    x = 'VSp_m3'
#    y = 'WP_multiplier_max'
#    z = 'PV_A_brutto'

    # New names
    # WP vs. PV
    x = 'P_th_WP_max_kW'
    y = 'P_el_PV_kW'
    z = 'VSp_m3'

    # WP vs. EH
#    x = 'P_th_WP_max_kW'
#    y = 'E_el_EH_max_kW'
#    z = 'VSp_m3'

    # WP vs VSp
#    x = 'VSp_m3'
#    y = 'P_th_WP_max_kW'
#    z = 'A_brutto_PV'

    # WP vs. PV
#    z = 'P_el_PV_kW'
#    y = 'P_el_SN2WP_max_kW'
#    x = 'VSp_m3'

#    x = 'VSp_m3'
#    y = 'P_th_WP_max_kW'
#    y = 'A_brutto_PV'
#    x = 's_iso_ref_m'
#    y = 'error'

    # Create plot with Bokeh
#    make_bokeh_colorbar(df, x, y, 'error')

    x_label = x
    y_label = y
    z_label = z

    # WP vs. PV
    x_label = r'$P_{th,WP}$ [kW]'
    y_label = r'$P_{el,PV}$ [kWp]'

    # WP vs. PV
#    x_label = r'$P_{th,WP}$ [kW]'
#    y_label = r'$P_{el,EH}$ [kWp]'

    # WP vs VSp
#    x_label = r'$V_{Sp}$ [m³]'
#    y_label = r'$P_{th,WP}$ [kW]'

    # WP vs. PV
#    x_label = r'$P_{el,PV}$ [kWp]'
#    y_label = r'$P_{el,SN,WP,max}$ [kW]'

#    y_label = r'$P_{th,WP,max}$ [kW]'
#    y_label = r'$A_{brutto,PV}$ [m²]'
#    z_label = r'$V_{Sp}$ [m³]'

#    build_gif(df, x, y, x_label=x_label, y_label=y_label)

    fig, ax = plot_contour(df[x], df[y], df['error'], x_label=x_label,
                           y_label=y_label, z_label='Fehler',
                           limits_xy=(min(df[x]), max(df[x]), min(df[y]), max(df[y])))

#    print(df)
    limit_error = 0.01
#    limit_error = 0.002
#    limit_error = 0.8
#    limit_error = 1

    df = df[df['error'] < limit_error]
#    df = df[df['P_th_WP_max_kW'] < 20000]
#    df = df[df['P_th_WP_max_kW'] < 1200]
#    df = df[df['E_el_EH_max_kW'] < 1100]
#    df = df[df['P_th_WP_max_kW'] < 1200]
#    df = df[df['A_brutto_PV'] < 50000]
#    df = df[df['P_el_PV_kW'] < 5500]
#    df = df[df['VSp_m3'] > 100]
#    print(df.head())
#    print(df)
    print(len(df))

    # Make a fit
#    df.sort_values(by=x, inplace=True)
#    func = poly2d_fit_and_plot(df[x], df[y], df[z], order=2,
#    func = poly_fit_and_plot(df[x], df[y], z=df[z], order=2,
    func = poly_fit_and_plot(df[x], df[y], order=-2,
                             x_label=x_label, y_label=y_label, z_label=z_label,
                             nx=30, ny=30,
#                             title='Points with error < '+str(limit_error),
                             title='',
                             show_plot=True,
                             print_formula=True,
                             input_fig_ax=(fig, ax),
#                             limits_xy=(min(df[x]), max(df[x]), min(df[y]), max(df[y])),
                             )
#    df['calc'] = func(df[x], df[y])

    # Use the fitted function to compute results
    dck_proc = trnpy.DCK_processor()
    param_table = dck_proc.parametric_table_combine({
#        'VSp_m3': [5500, 6000, 10000, 14000, 18000],
#        'VSp_m3': [60, 250, 500, 1000, 1500, 2000],
#        'P_th_WP_max_kW': [2000],
#        'VSp_m3': [50000],
#        'P_th_WP_max_kW': [5500, 6000, 10000, 14000, 18000],
        'P_th_WP_max_kW': [1600, 1700, 1800, 2000, 2350, 3000, 4000, 5000],
#        'P_th_WP_max_kW': [1250, 1400] + list(range(1650, 7000, 500)),
#        'P_th_WP_max_kW': [7000, 8000] + list(range(9000, 20000, 2000)),
#        'P_th_WP_max_kW': list(range(1250, 2501, 250)),
#        'A_brutto_PV': [20000, 24800, 32000, 36000, 40000],
#        'P_el_PV_kW': list(range(0, 5001, 1000)) + [5530],
#        'WP_multiplier_max': [45, 50, 55, 60, 65, ],
#        'P_th_WP_max_kW': [4100, 4500, 5000, 5500, 6000, 6500, 7000],
        })
#    param_table[z] = func(param_table[x], param_table[y], check_confidence=True)
#    param_table[y] = func(param_table[x], check_confidence=True)
    param_table[y] = func(param_table[x], check_confidence=False)

    param_table.dropna(axis='index', inplace=True)
    param_table = param_table.round(decimals=0)
    print(param_table)
    file = os.path.join(r'C:\Trnsys17\Work\futureSuN\AP4\P2H_Quartier\!Parameter_fit.xlsx')
    param_table.to_excel(file, index=False)

    # Use the fitted function to create a nomograph
#    make_nomograph(func)
