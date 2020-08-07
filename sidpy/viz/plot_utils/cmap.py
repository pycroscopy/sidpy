# -*- coding: utf-8 -*-
"""
Utilities for generating static image and line plots of near-publishable quality

Created on Thu May 05 13:29:12 2016

@author: Suhas Somnath, Chris R. Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals
from numbers import Number
import sys
import matplotlib as mpl
import numpy as np
from dask import array as da
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

if sys.version_info.major == 3:
    unicode = str

default_cmap = plt.cm.viridis


def get_cmap_object(cmap):
    """
    Get the matplotlib.colors.LinearSegmentedColormap object regardless of the input

    Parameters
    ----------
    cmap : String, or matplotlib.colors.LinearSegmentedColormap object (Optional)
        Requested color map
    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap object
        Requested / Default colormap object
    """
    if cmap is None:
        return default_cmap
    elif type(cmap) in [str, unicode]:
        return plt.get_cmap(cmap)
    elif not isinstance(cmap, mpl.colors.Colormap):
        raise TypeError('cmap should either be a matplotlib.colors.Colormap object or a string')
    return cmap


def cmap_jet_white_center():
    """
    Generates the jet colormap with a white center

    Returns
    -------
    white_jet : matplotlib.colors.LinearSegmentedColormap object
        color map object that can be used in place of the default colormap
    """
    # For red - central column is like brightness
    # For blue - last column is like brightness
    cdict = {'red': ((0.00, 0.0, 0.0),
                     (0.30, 0.0, 0.0),
                     (0.50, 1.0, 1.0),
                     (0.90, 1.0, 1.0),
                     (1.00, 0.5, 1.0)),
             'green': ((0.00, 0.0, 0.0),
                       (0.10, 0.0, 0.0),
                       (0.42, 1.0, 1.0),
                       (0.58, 1.0, 1.0),
                       (0.90, 0.0, 0.0),
                       (1.00, 0.0, 0.0)),
             'blue': ((0.00, 0.0, 0.5),
                      (0.10, 1.0, 1.0),
                      (0.50, 1.0, 1.0),
                      (0.70, 0.0, 0.0),
                      (1.00, 0.0, 0.0))
             }
    return LinearSegmentedColormap('white_jet', cdict)


def cmap_from_rgba(name, interp_vals, normalization_val):
    """
    Generates a colormap given a matlab-style interpolation table

    Parameters
    ----------
    name : String / Unicode
        Name of the desired colormap
    interp_vals : List of tuples
        Interpolation table that describes the desired color map. Each entry in the table should be described as:
        (position in the colorbar, (red, green, blue, alpha))
        The position in the color bar, red, green, blue, and alpha vary from 0 to the normalization value
    normalization_val : number
        The common maximum value for the position in the color bar, red, green, blue, and alpha

    Returns
    -------
    new_cmap : matplotlib.colors.LinearSegmentedColormap object
        desired color map
    """
    if not isinstance(name, (str, unicode)):
        raise TypeError('name should be a string')
    if not isinstance(interp_vals, (list, tuple, np.array)):
        raise TypeError('interp_vals must be a list of tuples')
    if not isinstance(normalization_val, Number):
        raise TypeError('normalization_val must be a number')

    normalization_val = np.round(1.0 * normalization_val)

    cdict = {'red': tuple([(dist / normalization_val, colors[0] / normalization_val, colors[0] / normalization_val)
                           for (dist, colors) in interp_vals][::-1]),
             'green': tuple([(dist / normalization_val, colors[1] / normalization_val, colors[1] / normalization_val)
                             for (dist, colors) in interp_vals][::-1]),
             'blue': tuple([(dist / normalization_val, colors[2] / normalization_val, colors[2] / normalization_val)
                            for (dist, colors) in interp_vals][::-1]),
             'alpha': tuple([(dist / normalization_val, colors[3] / normalization_val, colors[3] / normalization_val)
                             for (dist, colors) in interp_vals][::-1])}

    return LinearSegmentedColormap(name, cdict)


def make_linear_alpha_cmap(name, solid_color, normalization_val, min_alpha=0, max_alpha=1):
    """
    Generates a transparent to opaque color map based on a single solid color

    Parameters
    ----------
    name : String / Unicode
        Name of the desired colormap
    solid_color : List of numbers
        red, green, blue, and alpha values for a specific color
    normalization_val : number
        The common maximum value for the red, green, blue, and alpha values. This is 1 in matplotlib
    min_alpha : float (optional. Default = 0 : ie- transparent)
        Lowest alpha value for the bottom of the color bar
    max_alpha : float (optional. Default = 1 : ie- opaque)
        Highest alpha value for the top of the color bar

    Returns
    -------
    new_cmap : matplotlib.colors.LinearSegmentedColormap object
        transparent to opaque color map based on the provided color
    """
    if not isinstance(name, (str, unicode)):
        raise TypeError('name should be a string')
    if not isinstance(solid_color, (list, tuple, np.ndarray, da.core.Array)):
        raise TypeError('solid_color must be a list of numbers')
    if not len(solid_color) == 4:
        raise ValueError('solid-color should have fourth values')
    if not np.all([isinstance(x, Number) for x in solid_color]):
        raise TypeError('solid_color should have three numbers for red, green, blue')
    if not isinstance(normalization_val, Number):
        raise TypeError('normalization_val must be a number')
    if not isinstance(min_alpha, Number):
        raise TypeError('min_alpha should be a Number')
    if not isinstance(max_alpha, Number):
        raise TypeError('max_alpha should be a Number')
    if min_alpha >= max_alpha:
        raise ValueError('min_alpha must be less than max_alpha')

    solid_color = np.array(solid_color) / normalization_val * 1.0
    interp_table = [(1.0, (solid_color[0], solid_color[1], solid_color[2], max_alpha)),
                    (0, (solid_color[0], solid_color[1], solid_color[2], min_alpha))]
    return cmap_from_rgba(name, interp_table, 1)


def cmap_hot_desaturated():
    """
    Returns a desaturated color map based on the hot colormap

    Returns
    -------
    new_cmap : matplotlib.colors.LinearSegmentedColormap object
        Desaturated version of the hot color map
    """
    hot_desaturated = [(255.0, (255, 76, 76, 255)),
                       (218.5, (107, 0, 0, 255)),
                       (182.1, (255, 96, 0, 255)),
                       (145.6, (255, 255, 0, 255)),
                       (109.4, (0, 127, 0, 255)),
                       (72.675, (0, 255, 255, 255)),
                       (36.5, (0, 0, 91, 255)),
                       (0, (71, 71, 219, 255))]

    return cmap_from_rgba('hot_desaturated', hot_desaturated, 255)


def discrete_cmap(num_bins, cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map specified

    Parameters
    ----------
    num_bins : unsigned int
        Number of discrete bins
    cmap : matplotlib.colors.Colormap object
        Base color map to discretize

    Returns
    -------
    new_cmap : matplotlib.colors.LinearSegmentedColormap object
        Discretized color map

    Notes
    -----
    Jake VanderPlas License: BSD-style
    https://gist.github.com/jakevdp/91077b0cae40f8f8244a

    """
    if cmap is None:
        cmap = default_cmap.name

    elif isinstance(cmap, mpl.colors.Colormap):
        cmap = cmap.name
    elif not isinstance(cmap, (str, unicode)):
        raise TypeError('cmap should be a string or a matplotlib.colors.Colormap object')

    if not isinstance(num_bins, int):
        raise TypeError('num_bins must be an unsigned integer')

    return plt.get_cmap(cmap, num_bins)