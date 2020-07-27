# -*- coding: utf-8 -*-
"""
Utilities that assist in writing scientific data to HDF5 files

Created on Thu Sep  7 21:14:25 2017

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import sys
from itertools import groupby
import numpy as np
if sys.version_info.major == 3:
    from collections.abc import Iterable
else:
    from collections import Iterable

__all__ = ['clean_string_att', 'get_slope', 'to_ranges']

if sys.version_info.major == 3:
    unicode = str


def clean_string_att(att_val):
    """
    Replaces any unicode objects within lists with their string counterparts to ensure compatibility with python 3.
    If the attribute is indeed a list of unicodes, the changes will be made in-place

    Parameters
    ----------
    att_val : object
        Attribute object

    Returns
    -------
    att_val : object
        Attribute object
    """
    try:
        if isinstance(att_val, Iterable):
            if type(att_val) in [unicode, str]:
                return att_val
            elif np.any([type(x) in [str, unicode, bytes, np.str_] for x in att_val]):
                return np.array(att_val, dtype='S')
        if type(att_val) == np.str_:
            return str(att_val)
        return att_val
    except TypeError:
        raise TypeError('Failed to clean: {}'.format(att_val))


def get_slope(values, tol=1E-3):
    """
    Attempts to get the slope of the provided values. This function will be handy
    for checking if a dimension has been varied linearly or not.
    If the values vary non-linearly, a ValueError will be raised

    Parameters
    ----------
    values : array-like
        List of numbers
    tol : float, optional. Default = 1E-3
        Tolerance in the variation of the slopes.
    Returns
    -------
    float
        Slope of the line
    """
    if not isinstance(tol, float):
        raise TypeError('tol should be a float << 1')
    step_size = np.unique(np.diff(values))
    if len(step_size) > 1:
        # often we end up here. In most cases,
        step_avg = step_size.max()
        step_size -= step_avg
        var = np.mean(np.abs(step_size))
        if var / step_avg < tol:
            step_size = [step_avg]
        else:
            # Non-linear dimension! - see notes above
            raise ValueError('Provided values cannot be expressed as a linear trend')
    return step_size[0]


def to_ranges(iterable):
    """
    Converts a sequence of iterables to range tuples

    From https://stackoverflow.com/questions/4628333/converting-a-list-of-integers-into-range-in-python

    Credits: @juanchopanza and @luca

    Parameters
    ----------
    iterable : collections.Iterable object
        iterable object like a list

    Returns
    -------
    iterable : generator object
        Cast to list or similar to use
    """
    iterable = sorted(set(iterable))
    for key, group in groupby(enumerate(iterable), lambda t: t[1] - t[0]):
        group = list(group)
        if sys.version_info.major == 3:
            yield range(group[0][1], group[-1][1]+1)
        else:
            yield xrange(group[0][1], group[-1][1]+1)

