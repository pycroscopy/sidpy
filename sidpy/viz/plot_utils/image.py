# -*- coding: utf-8 -*-
"""
Utilities for generating static image and line plots of near-publishable quality

Created on Thu May 05 13:29:12 2016

@author: Suhas Somnath, Chris R. Smith
"""

from __future__ import division, print_function, absolute_import, \
    unicode_literals
import inspect
import sys
from numbers import Number
import matplotlib as mpl
import numpy as np
from dask import array as da
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from sidpy.base.num_utils import get_exponent
from sidpy.viz.plot_utils.misc import get_plot_grid_size, set_tick_font_size
from sidpy.viz.plot_utils.cmap import default_cmap

if sys.version_info.major == 3:
    unicode = str


def plot_map(axis, img, show_xy_ticks=True, show_cbar=True, x_vec=None, y_vec=None,
             num_ticks=4, stdevs=None, cbar_label=None, tick_font_size=None, infer_aspect=False, **kwargs):
    """
    Plots an image within the given axis with a color bar + label and appropriate X, Y tick labels.
    This is particularly useful to get readily interpretable plots for papers

    Parameters
    ----------
    axis : matplotlib.axes.Axes object
        Axis to plot this image onto
    img : 2D numpy array with real values
        Data for the image plot
    show_xy_ticks : bool, Optional, default = None, shown unedited
        Whether or not to show X, Y ticks
    show_cbar : bool, optional, default = True
        Whether or not to show the colorbar
    x_vec : 1-D array-like or Number, optional
        if an array-like is provided, these will be used for the tick values on the X axis
        if a Number is provided, this will serve as an extent for tick values in the X axis.
        For example x_vec=1.5 would cause the x tick labels to range from 0 to 1.5
    y_vec : 1-D array-like or Number, optional
        if an array-like is provided - these will be used for the tick values on the Y axis
        if a Number is provided, this will serve as an extent for tick values in the Y axis.
        For example y_vec=225 would cause the y tick labels to range from 0 to 225
    num_ticks : unsigned int, optional, default = 4
        Number of tick marks on the X and Y axes
    stdevs : unsigned int (Optional. Default = None)
        Number of standard deviations to consider for plotting.  If None, full range is plotted.
    cbar_label : str, optional, default = None
        Labels for the colorbar. Use this for something like quantity (units)
    tick_font_size : unsigned int, optional, default = None
        Font size to apply to x, y, colorbar ticks and colorbar label
    infer_aspect : bool, Optional. Default = False
        Whether or not to adjust the aspect ratio of the image based on the provided x_vec and y_vec
        The values of x_vec and y_vec will be assumed to have the same units.
    kwargs : dictionary
        Anything else that will be passed on to matplotlib.pyplot.imshow

    Returns
    -------
    im_handle : handle to image plot
        handle to image plot
    cbar : handle to color bar
        handle to color bar

    Note
    ----
    The origin of the image will be set to the lower left corner. Use the kwarg 'origin' to change this

    """
    if not isinstance(axis, mpl.axes.Axes):
        raise TypeError('axis must be a matplotlib.axes.Axes object')
    if not isinstance(img, (np.ndarray, da.core.Array)):
        raise TypeError('img should be a numpy array')
    if not img.ndim == 2:
        raise ValueError('img should be a 2D array')
    if not isinstance(show_xy_ticks, bool):
        raise TypeError('show_xy_ticks should be a boolean value')
    if not isinstance(show_cbar, bool):
        raise TypeError('show_cbar should be a boolean value')
    # checks for x_vec and y_vec are done below
    if num_ticks is not None:
        if not isinstance(num_ticks, int):
            raise TypeError('num_ticks should be a whole number')
        if num_ticks < 2:
            raise ValueError('num_ticks should be at least 2')
    if tick_font_size is not None:
        if not isinstance(tick_font_size, Number):
            raise TypeError('tick_font_size must be a whole number')
        if tick_font_size < 0:
            raise ValueError('tick_font_size must be a whole number')
    if stdevs is not None:
        if not isinstance(stdevs, Number):
            raise TypeError('stdevs should be a Number')
        data_mean = np.mean(img)
        data_std = np.std(img)
        kwargs.update({'clim': [data_mean - stdevs * data_std, data_mean + stdevs * data_std]})

    kwargs.update({'origin': kwargs.pop('origin', 'lower')})

    if show_cbar:
        
        if np.isnan(img).any():
            _img = img[np.where(~np.isnan(img))]
            y_exp = get_exponent(np.squeeze(_img))
        else:
            y_exp = get_exponent(np.squeeze(img))
        z_suffix = ''
        if y_exp < -2 or y_exp > 3:
            img = np.squeeze(img) / 10 ** y_exp
            z_suffix = ' x $10^{' + str(y_exp) + '}$'

    assert isinstance(show_xy_ticks, bool)

    ########################################################################################################

    def set_ticks_for_axis(tick_vals, is_x):
        if is_x:
            tick_vals_var_name = 'x_vec'
            tick_set_func = axis.set_xticks
            tick_labs_set_func = axis.set_xticklabels
        else:
            tick_vals_var_name = 'y_vec'
            tick_set_func = axis.set_yticks
            tick_labs_set_func = axis.set_yticklabels

        img_axis = int(is_x)
        img_size = img.shape[img_axis]
        chosen_ticks = np.linspace(0, img_size - 1, num_ticks, dtype=int)

        if tick_vals is not None:
            if isinstance(tick_vals, (int, float)):
                if tick_vals > 0.01:
                    tick_labs = [str(np.round(ind * tick_vals / img_size, 2)) for ind in chosen_ticks]
                else:
                    tick_labs = ['{0:.1E}'.format(ind * tick_vals / img_size) for ind in chosen_ticks]
                    print(tick_labs)
                tick_vals = np.linspace(0, tick_vals, img_size)
            else:
                if not isinstance(tick_vals, (np.ndarray, list, tuple, range, da.core.Array)) or \
                        len(tick_vals) != img_size:
                    raise ValueError(
                        '{} should be array-like with shape equal to axis {} of img'.format(tick_vals_var_name,
                                                                                            img_axis))
                if np.max(tick_vals) > 0.01:
                    tick_labs = [str(np.round(tick_vals[ind], 2)) for ind in chosen_ticks]
                else:
                    tick_labs = ['{0:.1E}'.format(tick_vals[ind]) for ind in chosen_ticks]
        else:
            tick_labs = [str(ind) for ind in chosen_ticks]

        tick_set_func(chosen_ticks)
        tick_labs_set_func(tick_labs)

        if tick_font_size is not None:
            set_tick_font_size(axis, tick_font_size)

        return tick_vals

    ########################################################################################################

    if show_xy_ticks is True or x_vec is not None:
        x_vec = set_ticks_for_axis(x_vec, True)
    else:
        axis.set_xticks([])

    if show_xy_ticks is True or y_vec is not None:
        y_vec = set_ticks_for_axis(y_vec, False)
    else:
        axis.set_yticks([])

    if infer_aspect:
        # Aspect ratio determined by this function will take precedence.
        _ = kwargs.pop('infer_aspect', None)

        """
        At this stage, if x_vec and y_vec are not None, they should be arrays.
        
        This will be very useful when one dimension is coarsely sampled while another is finely sampled
        and we want to visualize the image with the physically correct aspect ratio.
        This CANNOT be performed automatically due to potentially incompatible units which are unknown to this func.
        """

        if x_vec is not None or y_vec is not None:
            x_range = x_vec.max() - x_vec.min()
            y_range = y_vec.max() - y_vec.min()
            kwargs.update({'aspect': (y_range / x_range) * (img.shape[1] / img.shape[0])})

    im_handle = axis.imshow(img, **kwargs)

    cbar = None
    if not isinstance(show_cbar, bool):
        show_cbar = False

    if show_cbar:
        cbar = plt.colorbar(im_handle, ax=axis, orientation='vertical',
                            fraction=0.046, pad=0.04, use_gridspec=True)
        # cbar = axis.cbar_axes[count].colorbar(im_handle)

        if cbar_label is not None:
            if not isinstance(cbar_label, (str, unicode)):
                raise TypeError('cbar_label should be a string')

            if tick_font_size is not None:
                cbar.set_label(cbar_label + z_suffix)
            else:
                cbar.set_label(cbar_label + z_suffix, fontsize=tick_font_size)
        else:
            if z_suffix != '':
                cbar.set_label(z_suffix)

        if tick_font_size is not None:
            cbar.ax.tick_params(labelsize=tick_font_size)
    return im_handle, cbar


def plot_map_stack(map_stack, num_comps=9, stdevs=2, color_bar_mode=None, evenly_spaced=False, reverse_dims=False,
                   subtitle='Component', title=None, colorbar_label='', fig_mult=(5, 5), pad_mult=(0.1, 0.07),
                   x_label=None, y_label=None, title_yoffset=None, title_size=None, **kwargs):
    """
    Plots the provided stack of maps

    Parameters
    -------------
    map_stack : 3D real numpy array
        structured as [component, rows, cols]
    num_comps : int, Optional
        Number of components to plot
    stdevs : int, Optional
        Number of standard deviations to consider for plotting. Set to None if no clipping is desired
    color_bar_mode : String, Optional
        Options are None, single or each. Default None
    evenly_spaced : bool, Optional. Default = False
        If set to True - The slices / component will be selected at intervals from the first to last
        If set to False - The first ``num_comps`` images will be plotted instead
    reverse_dims : bool, Optional. Default = False
        Set this to True to accept data structured as [rows, cols, component]
    subtitle : String or list of strings
        The titles for each of the plots.
        If a single string is provided, the plot titles become ['title 01', title 02', ...].
        if a list of strings (equal to the number of components) are provided, these are used instead.
    title : str, Optinal
        Title for the plot grid that will appear at the top
    colorbar_label : str, Optional
        label for colorbar. Default is an empty string.
    fig_mult : length 2 array_like of uints
        Size multipliers for the figure.  Figure size is calculated as (num_rows*`fig_mult[0]`, num_cols*`fig_mult[1]`).
        Default (4, 4)
    pad_mult : tuple, list, array-like, Optional
        Array-like of floats of length 2.
        Multipliers for the axis padding between plots in the stack.  Padding is calculated as
        (pad_mult[0]*fig_mult[1], pad_mult[1]*fig_mult[0]) for the width and height padding respectively.
        Default (0.1, 0.07)
    x_label : str, Optional
        X Label for all plots
    y_label : (optional) String
        Y label for all plots
    title_yoffset : float
        Offset to move the figure title vertically in the figure
    title_size : float
        Size of figure title
    kwargs : dictionary
        Keyword arguments to be passed to either matplotlib.pyplot.figure, mpl_toolkits.axes_grid1.ImageGrid, or
        pyUSID.viz.plot_utils.plot_map.  See specific function documentation for the relavent options.

    Returns
    ---------
    fig, axes
    """
#    plt.rcParams["mpl_toolkits.legacy_colorbar"] = False

    if not isinstance(map_stack, (np.ndarray, da.core.Array)) or not map_stack.ndim == 3:
        raise TypeError('map_stack should be a 3 dimensional array arranged as [component, row, col]')
    if num_comps is None:
        num_comps = 4  # Default
    else:
        if not isinstance(num_comps, int) or num_comps < 1:
            raise TypeError('num_comps should be a positive integer')

    for var, var_name in zip([title, colorbar_label, color_bar_mode, x_label, y_label],
                             ['title', 'colorbar_label', 'color_bar_mode', 'x_label', 'y_label']):
        if var is not None:
            if not isinstance(var, (str, unicode)):
                raise TypeError(var_name + ' should be a string')

    if title is None:
        title = ''
    if colorbar_label is None:
        colorbar_label = ''
    if x_label is None:
        x_label = ''
    if y_label is None:
        y_label = ''

    if color_bar_mode not in [None, 'single', 'each']:
        raise ValueError('color_bar_mode must be either None, "single", or "each"')
    for var, var_name in zip([stdevs, title_yoffset, title_size],
                             ['stdevs', 'title_yoffset', 'title_size']):
        if var is not None:
            if not isinstance(var, Number) or var <= 0:
                raise TypeError(var_name + ' of value: {} should be a number > 0'.format(var))
    for var, var_name in zip([evenly_spaced, reverse_dims], ['evenly_spaced', 'reverse_dims']):
        if not isinstance(var, bool):
            raise TypeError(var_name + ' should be a bool')
    for var, var_name in zip([fig_mult, pad_mult], ['fig_mult', 'pad_mult']):
        if not isinstance(var, (list, tuple, np.ndarray, da.core.Array)) or len(var) != 2:
            raise TypeError(var_name + ' should be a tuple / list / numpy array of size 2')
        if not np.all([x > 0 and isinstance(x, Number) for x in var]):
            raise ValueError(var_name + ' should contain positive numbers')

    if reverse_dims:
        map_stack = np.transpose(map_stack, (2, 0, 1))

    num_comps = abs(num_comps)
    num_comps = min(num_comps, map_stack.shape[0])

    if evenly_spaced:
        chosen_pos = np.linspace(0, map_stack.shape[0] - 1, num_comps, dtype=int)
    else:
        chosen_pos = np.arange(num_comps, dtype=int)

    if isinstance(subtitle, list):

        if len(subtitle) > num_comps:
            # remove additional subtitles
            subtitle = subtitle[:num_comps]
        elif len(subtitle) < num_comps:
            # add subtitles
            subtitle += ['Component' + ' ' + str(x) for x in range(len(subtitle), num_comps)]
    else:
        if not isinstance(subtitle, str):
            subtitle = 'Component'
        subtitle = [subtitle + ' ' + str(x) for x in chosen_pos]

    fig_h, fig_w = fig_mult

    p_rows, p_cols = get_plot_grid_size(num_comps)

    if p_rows * p_cols < num_comps:
        p_cols += 1

    pad_w, pad_h = pad_mult

    '''
    Set defaults for kwargs to the figure creation and extract any non-default values from current kwargs
    '''
    figkwargs = dict()

    if sys.version_info.major == 3:
        inspec_func = inspect.getfullargspec
    else:
        inspec_func = inspect.getargspec

    for key in inspec_func(plt.figure).args:
        if key in kwargs:
            figkwargs.update({key: kwargs.pop(key)})

    fig = plt.figure(figsize=(p_cols * fig_w, p_rows * fig_h), **figkwargs)

    '''
    Set defaults for kwargs to the ImageGrid and extract any non-default values from current kwargs
    '''
    igkwargs = {'cbar_pad': '1%',
                'cbar_size': '5%',
                'cbar_location': 'right',
                'direction': 'row',
                'share_all': False,
                'aspect': True,
                'label_mode': 'L'}
    #            'add_all': True}
    for key in igkwargs.keys():
        if key in kwargs:
            igkwargs.update({key: kwargs.pop(key)})

    axes = ImageGrid(fig=fig, rect=111, nrows_ncols=(p_rows, p_cols),
                     cbar_mode=color_bar_mode,
                     axes_pad=(pad_w * fig_w, pad_h * fig_h),
                     **igkwargs)

    try:
        fig.canvas.set_window_title(title)
    except:
        fig.canvas.manager.set_window_title(title)
    # These parameters have not been easy to fix:
    if title_yoffset is None:
        title_yoffset = 0.9
    if title_size is None:
        title_size = 16 + (p_rows + p_cols)
    fig.suptitle(title, fontsize=title_size, y=title_yoffset)
#    plt.rcParams["mpl_toolkits.legacy_colorbar"] = False

    for count, index, curr_subtitle in zip(range(chosen_pos.size), chosen_pos, subtitle):
        im, im_cbar = plot_map(axes[count],
                               map_stack[index],
                               stdevs=stdevs, show_cbar=False, **kwargs)
        axes[count].set_title(curr_subtitle)

        if color_bar_mode == 'each':
            cb = axes[count].cax.colorbar(im)
            if count % p_cols == p_cols-1:
                cb.set_label(colorbar_label)

        if count % p_cols == 0:
            axes[count].set_ylabel(y_label)
        if count >= (p_rows - 1) * p_cols:
            axes[count].set_xlabel(x_label)

    # With cbar_mode="single", cax attribute of all axes are identical.
    if color_bar_mode == 'single':
        cb = axes[0].cax.colorbar(im)
        cb.set_label(colorbar_label)

    return fig, axes
