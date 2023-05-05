# -*- coding: utf-8 -*-
"""
Utilities for generating static image and line plots of near-publishable quality

Created on Thu May 05 13:29:12 2016

@author: Suhas Somnath, Chris R. Smith
"""

from __future__ import division, print_function, absolute_import, \
    unicode_literals
from numbers import Number
import sys
import h5py
import matplotlib as mpl
import numpy as np
from dask import array as da
from matplotlib import pyplot as plt

from sidpy.viz.plot_utils.misc import get_plot_grid_size, make_scalar_mappable
from sidpy.viz.plot_utils.cmap import default_cmap, get_cmap_object, discrete_cmap
from sidpy.viz.plot_utils.image import plot_map

if sys.version_info.major == 3:
    unicode = str


def cbar_for_line_plot(axis, num_steps, discrete_ticks=True, **kwargs):
    """
    Adds a colorbar next to a line plot axis

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis with multiple line objects
    num_steps : uint
        Number of steps in the colorbar
    discrete_ticks : (optional) bool
        Whether or not to have the ticks match the number of number of steps. Default = True
    """
    if not isinstance(axis, mpl.axes.Axes):
        raise TypeError('axis must be a matplotlib.axes.Axes object')
    if not isinstance(num_steps, int) and num_steps > 0:
        raise TypeError('num_steps must be a whole number')
    assert isinstance(discrete_ticks, bool)

    cmap = get_cmap_object(kwargs.pop('cmap', None))
    cmap = discrete_cmap(num_steps, cmap=cmap.name)

    sm = make_scalar_mappable(0, num_steps, cmap=cmap)

    if discrete_ticks:
        kwargs.update({'ticks': np.arange(num_steps)})

    cbar = plt.colorbar(sm, ax=axis, orientation='vertical',
                        pad=0.04, use_gridspec=True, **kwargs)
    return cbar


def rainbow_plot(axis, x_vec, y_vec, num_steps=32, **kwargs):
    """
    Plots the input against the output vector such that the color of the curve changes as a function of index

    Parameters
    ----------
    axis : matplotlib.axes.Axes object
        Axis to plot the curve
    x_vec : 1D float numpy array
        vector that forms the X axis
    y_vec : 1D float numpy array
        vector that forms the Y axis
    num_steps : unsigned int (Optional)
        Number of discrete color steps
    """
    if not isinstance(axis, mpl.axes.Axes):
        raise TypeError('axis must be a matplotlib.axes.Axes object')
    if not isinstance(x_vec, (list, tuple, np.ndarray, da.core.Array)):
        raise TypeError('x_vec must be array-like of numbers')
    if not isinstance(x_vec, (list, tuple, np.ndarray, da.core.Array)):
        raise TypeError('x_vec must be array-like of numbers')
    x_vec = np.array(x_vec)
    y_vec = np.array(y_vec)
    assert x_vec.ndim == 1 and y_vec.ndim == 1, 'x_vec and y_vec must be 1D arrays'
    assert x_vec.shape == y_vec.shape, 'x_vec and y_vec must have the same shape'

    if not isinstance(num_steps, int):
        raise TypeError('num_steps must be an integer < size of x_vec')
    if num_steps < 2 or num_steps >= len(x_vec) // 2:
        raise ValueError('num_steps should be a positive number. 1/4 to 1/16th of x_vec')
    assert num_steps < x_vec.size, 'num_steps must be an integer < size of x_vec'

    assert isinstance(kwargs, dict)
    cmap = kwargs.pop('cmap', default_cmap)
    cmap = get_cmap_object(cmap)

    # Remove any color flag
    _ = kwargs.pop('color', None)

    pts_per_step = len(y_vec) // num_steps

    for step in range(num_steps - 1):
        axis.plot(x_vec[step * pts_per_step:(step + 1) * pts_per_step],
                  y_vec[step * pts_per_step:(step + 1) * pts_per_step],
                  color=cmap(255 * step // num_steps), **kwargs)
    # plot the remainder:
    axis.plot(x_vec[(num_steps - 1) * pts_per_step:],
              y_vec[(num_steps - 1) * pts_per_step:],
              color=cmap(255 * num_steps / num_steps), **kwargs)


def plot_line_family(axis, x_vec, line_family, line_names=None, label_prefix='', label_suffix='',
                     y_offset=0, show_cbar=False, **kwargs):
    """
    Plots a family of lines with a sequence of colors

    Parameters
    ----------
    axis : matplotlib.axes.Axes object
        Axis to plot the curve
    x_vec : array-like
        Values to plot against
    line_family : 2D numpy array
        family of curves arranged as [curve_index, features]
    line_names : array-like
        array of string or numbers that represent the identity of each curve in the family
    label_prefix : string / unicode
        prefix for the legend (before the index of the curve)
    label_suffix : string / unicode
        suffix for the legend (after the index of the curve)
    y_offset : (optional) number
        quantity by which the lines are offset from each other vertically (useful for spectra)
    show_cbar : (optional) bool
        Whether or not to show a colorbar (instead of a legend)

    """
    if not isinstance(axis, mpl.axes.Axes):
        raise TypeError('axis must be a matplotlib.axes.Axes object')
    if not isinstance(x_vec, (list, tuple, np.ndarray, da.core.Array)):
        raise TypeError('x_vec must be array-like of numbers')
    x_vec = np.array(x_vec)
    assert x_vec.ndim == 1, 'x_vec must be a 1D array'
    if not isinstance(line_family, list):
        line_family = np.array(line_family)
    if not isinstance(line_family, (np.ndarray, da.core.Array)):
        raise TypeError('line_family must be a 2d array of numbers')
    assert line_family.ndim == 2, 'line_family must be a 2D array'
    #    assert x_vec.shape[1] == line_family.shape[1], \
    #        'The size of the 2nd dimension of line_family must match with of x_vec, but line fam has shape {} whereas xvec has shape {}'.format(line_family.shape, x_vec.shape)
    num_lines = line_family.shape[0]
    for var, var_name in zip([label_suffix, label_prefix], ['label_suffix', 'label_prefix']):
        if not isinstance(var, (str, unicode)):
            raise TypeError(var_name + ' needs to be a string')
    if not isinstance(y_offset, Number):
        raise TypeError('y_offset should be a Number')
    assert isinstance(show_cbar, bool)
    if line_names is not None:
        if not isinstance(line_names, (list, tuple)):
            raise TypeError('line_names should be a list of strings')
        if not np.all([isinstance(x, (str, unicode)) for x in line_names]):
            raise TypeError('line_names should be a list of strings')
        if len(line_names) != num_lines:
            raise ValueError('length of line_names not matching with that of line_family')

    cmap = get_cmap_object(kwargs.pop('cmap', None))

    if line_names is None:
        # label_prefix = 'Line '
        line_names = [str(line_ind) for line_ind in range(num_lines)]

    line_names = ['{} {} {}'.format(label_prefix, cur_name, label_suffix) for cur_name in line_names]

    for line_ind in range(num_lines):
        axis.plot(x_vec, line_family[line_ind] + line_ind * y_offset,
                  label=line_names[line_ind],
                  color=cmap(int(255 * line_ind / (num_lines ))), **kwargs)

    if show_cbar:
        # put back the cmap parameter:
        kwargs.update({'cmap': cmap})
        _ = cbar_for_line_plot(axis, num_lines, **kwargs)


def plot_curves(excit_wfms, datasets, line_colors=[], dataset_names=[], evenly_spaced=True,
                num_plots=25, x_label='', y_label='', subtitle_prefix='Position', title='',
                use_rainbow_plots=False, fig_title_yoffset=1.05, h5_pos=None, **kwargs):
    """
    Plots curves / spectras from multiple datasets from up to 25 evenly spaced positions
    Parameters
    -----------
    excit_wfms : 1D numpy float array or list of same
        Excitation waveform in the time domain
    datasets : list of 2D numpy arrays or 2D hyp5.Dataset objects
        Datasets containing data arranged as (pixel, time)
    line_colors : list of strings
        Colors to be used for each of the datasets
    dataset_names : (Optional) list of strings
        Names of the different datasets to be compared
    evenly_spaced : boolean
        Evenly spaced positions or first N positions
    num_plots : unsigned int
        Number of plots
    x_label : (optional) String
        X Label for all plots
    y_label : (optional) String
        Y label for all plots
    subtitle_prefix : (optional) String
        prefix for title over each plot
    title : (optional) String
        Main plot title
    use_rainbow_plots : (optional) Boolean
        Plot the lines as a function of spectral index (eg. time)
    fig_title_yoffset : (optional) float
        Y offset for the figure title. Value should be around 1
    h5_pos : HDF5 dataset reference or 2D numpy array
        Dataset containing position indices
    Returns
    ---------
    fig, axes

    """
    for var, var_name in zip([use_rainbow_plots, evenly_spaced], ['use_rainbow_plots', 'evenly_spaced']):
        if not isinstance(var, bool):
            raise TypeError(var_name + ' should be of type: bool')
    for var, var_name in zip([x_label, y_label, subtitle_prefix, title],
                             ['x_label', 'y_label', 'subtitle_prefix', 'title']):
        if var is not None:
            if not isinstance(var, (str, unicode)):
                raise TypeError(var_name + ' should be of type: str')
        else:
            var = ''

    if fig_title_yoffset is not None:
        if not isinstance(fig_title_yoffset, Number):
            raise TypeError('fig_title_yoffset should be a Number')
    else:
        fig_title_yoffset = 1.0

    if h5_pos is not None:
        if not isinstance(h5_pos, h5py.Dataset):
            raise TypeError('h5_pos should be a h5py.Dataset object')
    if not isinstance(num_plots, int) or num_plots < 1:
        raise TypeError('num_plots should be a number')

    for var, var_name, dim_size in zip([datasets, excit_wfms], ['datasets', 'excit_wfms'], [2, 1]):
        mesg = '{} should be {}D arrays or iterables (list or tuples) of {}D arrays' \
               '.'.format(var_name, dim_size, dim_size)
        if isinstance(var, (h5py.Dataset, np.ndarray, da.core.Array)):
            if not len(var.shape) == dim_size:
                raise ValueError(mesg)
        elif isinstance(var, (list, tuple)):
            if not np.all([isinstance(dset, (h5py.Dataset, np.ndarray, da.core.Array)) for dset in datasets]):
                raise TypeError(mesg)
        else:
            raise TypeError(mesg)

    # modes:
    # 0 = one excitation waveform and one dataset
    # 1 = one excitation waveform but many datasets
    # 2 = one excitation waveform for each of many dataset
    if isinstance(datasets, (h5py.Dataset, np.ndarray, da.core.Array)):
        # can be numpy array or h5py.dataset
        num_pos = datasets.shape[0]
        num_points = datasets.shape[1]
        datasets = [datasets]
        if isinstance(excit_wfms, (np.ndarray, h5py.Dataset, da.core.Array)):
            excit_wfms = [excit_wfms]
        elif isinstance(excit_wfms, list):
            if len(excit_wfms) == num_points:
                excit_wfms = [np.array(excit_wfms)]
            elif len(excit_wfms) == 1 and len(excit_wfms[0]) == num_points:
                excit_wfms = [np.array(excit_wfms[0])]
            else:
                raise ValueError('If only a single dataset is provided, excit_wfms should be a 1D array')
        line_colors = ['b']
        dataset_names = ['Default']
        mode = 0
    else:
        # dataset is a list of datasets
        # First check if the datasets are correctly shaped:
        num_pos_es = list()
        num_points_es = list()

        for dataset in datasets:
            if not isinstance(dataset, (h5py.Dataset, np.ndarray, da.core.Array)):
                raise TypeError('datasets can be a list of 2D h5py.Dataset or numpy array objects')
            if len(dataset.shape) != 2:
                raise ValueError('Each datset should be a 2D array')
            num_pos_es.append(dataset.shape[0])
            num_points_es.append(dataset.shape[1])

        num_pos_es = np.array(num_pos_es)
        num_points_es = np.array(num_points_es)

        if np.unique(num_pos_es).size > 1:  # or np.unique(num_points_es).size > 1:
            raise ValueError('The first dimension of the datasets are not matching: ' + str(num_pos_es))
        num_pos = np.unique(num_pos_es)[0]

        if len(excit_wfms) == len(datasets):
            # one excitation waveform per dataset but now verify each size
            if not np.all([len(cur_ex) == cur_dset.shape[1] for cur_ex, cur_dset in zip(excit_wfms, datasets)]):
                raise ValueError('Number of points in the datasets do not match with the excitation waveforms')
            mode = 2
        else:
            # one excitation waveform for all datasets
            if np.unique(num_points_es).size > 1:
                raise ValueError('Datasets don not contain the same number of points: ' + str(num_points_es))
            # datasets of the same size but does this match with the size of excitation waveforms:
            if len(excit_wfms) != np.unique(num_points_es)[0]:
                raise ValueError('Number of points in dataset not matching with shape of excitation waveform')
            excit_wfms = [excit_wfms]
            mode = 1

        for var, var_name in zip([dataset_names, line_colors], ['dataset_names', 'line_colors']):
            if not isinstance(var, (list, tuple)) or not np.all([isinstance(x, (str, unicode)) for x in var]):
                raise TypeError(var_name + ' should be a list of strings')
            if len(var) > 0 and len(var) != len(datasets):
                raise ValueError(var_name + ' is not of same length as datasets: ' + len(datasets))

        # Next the identification of datasets:
        if len(dataset_names) == 0:
            dataset_names = ['Dataset' + ' ' + str(x) for x in range(len(dataset_names), len(datasets))]

        if len(line_colors) == 0:
            # TODO: Generate colors from a user-specified colormap or consider using line family
            color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink', 'brown', 'orange']
            if len(datasets) < len(color_list):
                remaining_colors = [x for x in color_list if x not in line_colors]
                line_colors += remaining_colors[:len(datasets) - len(color_list)]
            else:
                raise ValueError('Insufficient number of line colors provided')

    # cannot support rainbows with multiple datasets!
    use_rainbow_plots = use_rainbow_plots and len(datasets) == 1

    if mode != 2:
        # convert it to something like mode 2
        excit_wfms = [excit_wfms[0] for _ in range(len(datasets))]

    if mode != 0:
        # users are not allowed to specify colors
        _ = kwargs.pop('color', None)

    num_plots = min(min(num_plots, 49), num_pos)
    nrows, ncols = get_plot_grid_size(num_plots)

    if evenly_spaced:
        chosen_pos = np.linspace(0, num_pos - 1, nrows * ncols, dtype=int)
    else:
        chosen_pos = np.arange(nrows * ncols, dtype=int)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(12, 12))
    if type(axes)==np.ndarray:
        axes_lin = axes.flatten()
    else: axes_lin = [axes]
    
    for count, posn in enumerate(chosen_pos):
        if use_rainbow_plots:
            rainbow_plot(axes_lin[count], excit_wfms[0], datasets[0][posn], **kwargs)
        else:
            for dataset, ex_wfm, col_val in zip(datasets, excit_wfms, line_colors):
                axes_lin[count].plot(ex_wfm, dataset[posn], color=col_val, **kwargs)
        if h5_pos is not None:
            # print('Row ' + str(h5_pos[posn,1]) + ' Col ' + str(h5_pos[posn,0]))
            # TODO: Do NOT assume 2 pos dims. Also format with low precision, use correct dim name, units as well
            axes_lin[count].set_title('Row ' + str(h5_pos[posn, 1]) + ' Col ' + str(h5_pos[posn, 0]), fontsize=12)
        else:
            axes_lin[count].set_title(subtitle_prefix + ' ' + str(posn), fontsize=12)

        if count % ncols == 0:
            axes_lin[count].set_ylabel(y_label, fontsize=12)
        if count >= (nrows - 1) * ncols:
            axes_lin[count].set_xlabel(x_label, fontsize=12)
        axes_lin[count].axis('tight')
        axes_lin[count].set_aspect('auto')
        axes_lin[count].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    if len(datasets) > 1:
        axes_lin[count].legend(dataset_names, loc='best')
    if title:
        fig.suptitle(title, fontsize=14, y=fig_title_yoffset)
    plt.tight_layout()
    return fig, axes


def plot_complex_spectra(map_stack, x_vec=None, num_comps=4, title=None, x_label='', y_label='', evenly_spaced=True,
                         subtitle_prefix='Component', amp_units=None, stdevs=2, **kwargs):
    """
    Plots the amplitude and phase components of the provided stack of complex valued spectrograms (2D images)

    Parameters
    -------------
    map_stack : 2D or 3D numpy complex matrices
        stack of complex valued 1D spectra arranged as [component, spectra] or
        2D images arranged as - [component, row, col]
    x_vec : 1D array-like, optional, default=None
        If the data are spectra (1D) instead of spectrograms (2D), x_vec is the reference array against which
    num_comps : int
        Number of images to plot
    title : str, optional
        Title to plot above everything else
    x_label : str, optional
        Label for x axis
    y_label : str, optional
        Label for y axis
    evenly_spaced : bool, optional. Default = True
        If True, images will be sampled evenly over the given dataset. Else, the first num_comps images will be plotted
    subtitle_prefix : str, optional
        Prefix for the title over each image
    amp_units : str, optional
        Units for amplitude
    stdevs : int
        Number of standard deviations to consider for plotting

    **kwargs will be passed on either to plot_map() or pyplot.plot()

    Returns
    ---------
    fig, axes
    """
    if not isinstance(map_stack, (np.ndarray, da.core.Array)) or not map_stack.ndim in [2, 3]:
        raise TypeError('map_stack should be a 2/3 dimensional array arranged as [component, row, col] or '
                        '[component, spectra')
    if x_vec is not None:
        if not isinstance(x_vec, (list, tuple, np.ndarray, da.core.Array)):
            raise TypeError('x_vec should be a 1D array')
        x_vec = np.array(x_vec)
        if x_vec.ndim != 1:
            raise ValueError('x_vec should be a 1D array')
        if x_vec.size != map_stack.shape[1]:
            raise ValueError('x_vec: {} should be of the same size as the second dimension of map_stack: '
                             '{}'.format(x_vec.shape, map_stack.shape))
    else:
        if map_stack.ndim == 2:
            x_vec = np.arange(map_stack.shape[1])

    if num_comps is None:
        num_comps = 4  # Default
    else:
        if not isinstance(num_comps, int) or not num_comps > 0:
            raise TypeError('num_comps should be a positive integer')
    for var, var_name in zip([title, x_label, y_label, subtitle_prefix, amp_units],
                             ['title', 'x_label', 'y_label', 'subtitle_prefix', 'amp_units']):
        if var is not None:
            if not isinstance(var, (str, unicode)):
                raise TypeError(var_name + ' should be a string')
    if amp_units is None:
        amp_units = 'a.u.'
    if stdevs is not None:
        if not isinstance(stdevs, Number) or stdevs <= 0:
            raise TypeError('stdevs should be a positive number')

    num_comps = min(24, min(num_comps, map_stack.shape[0]))

    if evenly_spaced:
        chosen_pos = np.linspace(0, map_stack.shape[0] - 1, num_comps, dtype=int)
    else:
        chosen_pos = np.arange(num_comps, dtype=int)

    nrows, ncols = get_plot_grid_size(num_comps)

    figsize = kwargs.pop('figsize', (4, 4))  # Individual plot size
    figsize = (figsize[0] * ncols, figsize[1] * nrows)

    fig, axes = plt.subplots(nrows * 2, ncols, figsize=figsize)
    fig.subplots_adjust(hspace=0.1, wspace=0.4)
    if title is not None:
        fig.canvas.manager.set_window_title(title)
        fig.suptitle(title, y=1.025)

    title_prefix = ''
    for comp_counter, comp_pos in enumerate(chosen_pos):
        ax_ind = (comp_counter // ncols) * (2 * ncols) + comp_counter % ncols
        cur_axes = [axes.flat[ax_ind], axes.flat[ax_ind + ncols]]
        funcs = [np.abs, np.angle]
        labels = ['Amplitude (' + amp_units + ')', 'Phase (rad)']
        for func, comp_name, axis, std_val in zip(funcs, labels, cur_axes, [stdevs, None]):
            y_vec = func(map_stack[comp_pos])
            if map_stack.ndim > 2:
                kwargs['stdevs'] = std_val
                _ = plot_map(axis, y_vec, **kwargs)
            else:
                axis.plot(x_vec, y_vec, **kwargs)

            if num_comps > 1:
                title_prefix = '%s %d - ' % (subtitle_prefix, comp_counter)
            axis.set_title('%s%s' % (title_prefix, comp_name))

            axis.set_aspect('auto')
            if ax_ind % ncols == 0:
                axis.set_ylabel(y_label)
            if np.ceil((ax_ind + ncols) / ncols) == nrows:
                axis.set_xlabel(x_label)

    fig.tight_layout()

    return fig, axes


def plot_scree(scree, title='Scree', **kwargs):
    """
    Plots the scree from SVD

    Parameters
    -------------
    scree : 1D real numpy array
        The scree vector from SVD
    title : str
        Figure title.  Default Scree

    Returns
    ---------
    fig, axes
    """
    if isinstance(scree, (list, tuple)):
        scree = np.array(scree)

    if not (isinstance(scree, (np.ndarray, da.core.Array)) or isinstance(scree, h5py.Dataset)):
        raise TypeError('scree must be a 1D array or Dataset')
    if not isinstance(title, (str, unicode)):
        raise TypeError('title must be a string')
    if h5py.__version__ >= '3' and isinstance(scree, h5py.Dataset):
        scree = scree[()]

    fig = plt.figure(figsize=kwargs.pop('figsize', (6.5, 6)))
    axis = fig.add_axes([0.1, 0.1, .8, .8])  # left, bottom, width, height (range 0 to 1)
    kwargs.update({'color': kwargs.pop('color', 'b')})
    kwargs.update({'marker': kwargs.pop('marker', '*')})
    axis.loglog(np.arange(len(scree)) + 1, scree, **kwargs)
    axis.set_xlabel('Component')
    axis.set_ylabel('Variance')
    axis.set_title(title)
    axis.set_xlim(left=1, right=len(scree))
    axis.set_ylim(bottom=np.min(scree), top=np.max(scree))
    fig.canvas.manager.set_window_title(title)

    return fig, axis
