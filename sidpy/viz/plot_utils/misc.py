# -*- coding: utf-8 -*-
"""
Utilities for generating static image and line plots of near-publishable quality

Created on Thu May 05 13:29:12 2016

@author: Suhas Somnath, Chris R. Smith
"""
from __future__ import division, print_function, absolute_import, unicode_literals
import os
import sys
from numbers import Number
import numpy as np
import matplotlib as mpl
from matplotlib import ticker as mtick, pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sidpy.viz.plot_utils.cmap import default_cmap

if sys.version_info.major == 3:
    unicode = str


def reset_plot_params():
    """
    Resets the plot parameters to matplotlib default values
    Adapted from:
    https://stackoverflow.com/questions/26413185/how-to-recover-matplotlib-defaults-after-setting-stylesheet
    """

    mpl.rcParams.update(mpl.rcParamsDefault)
    # Also resetting ipython inline parameters
    inline_rc = dict(mpl.rcParams)
    mpl.rcParams.update(inline_rc)


def use_nice_plot_params():
    """
    Resets default plot parameters such as figure size, font sizes etc. to values better suited for scientific
    publications
    """
    # mpl.rcParams.keys()  # gets all allowable keys
    # mpl.rc('figure', figsize=(5.5, 5))
    mpl.rc('lines', linewidth=2)
    mpl.rc('axes', labelsize=16, titlesize=16)
    mpl.rc('figure', titlesize=20)
    mpl.rc('font', size=14)  # global font size
    mpl.rc('legend', fontsize=16, fancybox=True)
    mpl.rc('xtick.major', size=6)
    mpl.rc('xtick.minor', size=4)
    # mpl.rcParams['xtick.major.size'] = 6


def set_tick_font_size(axes, font_size):
    """
    Sets the font size of the ticks in the provided axes

    Parameters
    ----------
    axes : matplotlib.pyplot.axis object or list of axis objects
        axes to set font sizes
    font_size : unigned int
        Font size
    """
    assert isinstance(font_size, Number)
    font_size = max(1, int(font_size))

    def __set_axis_tick(axis):
        """
        Sets the font sizes to the x and y axis in the given axis object

        Parameters
        ----------
        axis : matplotlib.axes.Axes object
            axis to set font sizes
        """
        for tick in axis.xaxis.get_major_ticks():
            tick.label1.set_fontsize(font_size)
        for tick in axis.yaxis.get_major_ticks():
            tick.label1.set_fontsize(font_size)

    mesg = 'axes must either be a matplotlib.axes.Axes object or an iterable containing such objects'

    if hasattr(axes, '__iter__'):
        for axis in axes:
            assert isinstance(axis, mpl.axes.Axes), mesg
            __set_axis_tick(axis)
    else:
        assert isinstance(axes, mpl.axes.Axes), mesg
        __set_axis_tick(axes)


def use_scientific_ticks(axis, is_x=True, formatting='%.2e'):
    """
    Makes the desired axis use scientific notation for its tick labels. This is applicable only for 1D plots at the
    moment.

    Parameters
    ----------
    axis : matplotlib.pyplot.axis object
        Axis handle
    is_x : bool, optional. Default = True
        If set to true, scientific notation will be applied only to the X axis.
        If set to False, scientific notation will be applied only to the Y axis.
    formatting : str / unicode, optional. Default = 2 digits of precision
        Precision for the tick labels
    """
    if not isinstance(axis, mpl.axes.Axes):
        raise TypeError('axis must be a matplotlib.axes.Axes object')
    if not isinstance(is_x, bool):
        raise TypeError('is_x should be a boolean to avoid confusion')
    if not isinstance(formatting, (str, unicode)):
        raise TypeError('formatting must be a string')

    if is_x:
        ax_hand = axis.xaxis
    else:
        ax_hand = axis.yaxis
    ax_hand.set_major_formatter(mtick.FormatStrFormatter(formatting))


def make_scalar_mappable(vmin, vmax, cmap=None):
    """
    Creates a scalar mappable object that can be used to create a colorbar for non-image (e.g. - line) plots

    Parameters
    ----------
    vmin : Number
        Minimum value for colorbar
    vmax : Number
        Maximum value for colorbar
    cmap : colormap object
        Colormap object to use

    Returns
    -------
    sm : matplotlib.pyplot.cm.ScalarMappable object
        The object that can used to create a colorbar via plt.colorbar(sm)

    Adapted from: https://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots
    """
    assert isinstance(vmin, Number), 'vmin should be a number'
    assert isinstance(vmax, Number), 'vmax should be a number'
    assert vmin < vmax, 'vmin must be less than vmax'

    if cmap is None:
        cmap = default_cmap
    else:
        assert isinstance(cmap, (mpl.colors.Colormap, str, unicode))
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # fake up the array of the scalar mappable
    sm._A = []
    return sm


def get_plot_grid_size(num_plots, fewer_rows=True):
    """
    Returns the number of rows and columns ideal for visualizing multiple (identical) plots within a single figure

    Parameters
    ----------
    num_plots : uint
        Number of identical subplots within a figure
    fewer_rows : bool, optional. Default = True
        Set to True if the grid should be short and wide or False for tall and narrow

    Returns
    -------
    nrows : uint
        Number of rows
    ncols : uint
        Number of columns
    """
    assert isinstance(num_plots, Number), 'num_plots must be a number'
    # force integer:
    num_plots = int(num_plots)
    if num_plots < 1:
        raise ValueError('num_plots was less than 0')

    if fewer_rows:
        nrows = int(np.floor(np.sqrt(num_plots)))
        ncols = int(np.ceil(num_plots / nrows))
    else:
        ncols = int(np.floor(np.sqrt(num_plots)))
        nrows = int(np.ceil(num_plots / ncols))

    return nrows, ncols


def export_fig_data(fig, filename, include_images=False):
    """
    Export the data of all plots in the figure `fig` to a plain text file.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure containing the data to be exported
    filename : str
        The filename of the output text file
    include_images : bool
        Should images in the figure also be exported

    Returns
    -------

    """
    # Get the data from the figure
    axes = fig.get_axes()
    axes_dict = dict()
    for ax in axes:
        ax_dict = dict()

        ims = ax.get_images()
        if len(ims) != 0 and include_images:
            im_dict = dict()

            for im in ims:
                # Image data
                im_lab = im.get_label()
                im_dict['data'] = im.get_array().data

                # X-Axis
                x_ax = ax.get_xaxis()
                x_lab = x_ax.label.get_label()
                if x_lab == '':
                    x_lab = 'X'

                im_dict[x_lab] = x_ax.get_data_interval()

                # Y-Axis
                y_ax = ax.get_yaxis()
                y_lab = y_ax.label.get_label()
                if y_lab == '':
                    y_lab = 'Y'

                im_dict[y_lab] = y_ax.get_data_interval()

                ax_dict['Images'] = {im_lab: im_dict}

        lines = ax.get_lines()
        if len(lines) != 0:
            line_dict = dict()

            xlab = ax.get_xlabel()
            ylab = ax.get_ylabel()

            if xlab == '':
                xlab = 'X Data'
            if ylab == '':
                ylab = 'Y Data'

            for line in lines:
                line_dict[line.get_label()] = {xlab: line.get_xdata(),
                                               ylab: line.get_ydata()}

            ax_dict['Lines'] = line_dict

        if ax_dict != dict():
            axes_dict[ax.get_title()] = ax_dict

    '''
    Now that we have the data from the figure, we need to write it to file.
    '''

    filename = os.path.abspath(filename)
    basename, ext = os.path.splitext(filename)
    folder, _ = os.path.split(basename)

    spacer = r'**********************************************\n'

    data_file = open(filename, 'w')

    data_file.write(fig.get_label() + '\n')
    data_file.write('\n')

    for ax_lab, ax in axes_dict.items():
        data_file.write('Axis: {} \n'.format(ax_lab))

        if 'Images' not in ax:
            continue
        for im_lab, im in ax['Images'].items():
            data_file.write('Image: {} \n'.format(im_lab))
            data_file.write('\n')
            im_data = im.pop('data')
            for row in im_data:
                row.tofile(data_file, sep='\t', format='%s')
                data_file.write('\n')
            data_file.write('\n')

            for key, val in im.items():
                data_file.write(key + '\n')

                val.tofile(data_file, sep='\n', format='%s')
                data_file.write('\n')

            data_file.write(spacer)

        if 'Lines' not in ax:
            continue
        for line_lab, line_dict in ax['Lines'].items():
            data_file.write('Line: {} \n'.format(line_lab))
            data_file.write('\n')

            dim1, dim2 = line_dict.keys()

            data_file.write('{} \t {} \n'.format(dim1, dim2))
            for val1, val2 in zip(line_dict[dim1], line_dict[dim2]):
                data_file.write('{} \t {} \n'.format(str(val1), str(val2)))

            data_file.write(spacer)

        data_file.write(spacer)

    data_file.close()