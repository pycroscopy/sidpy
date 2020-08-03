# -*- coding: utf-8 -*-
"""
Utilities that assist in dictionary manipulation

Created on Thu Sep  7 21:14:25 2017

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, unicode_literals, absolute_import
from collections.abc import MutableMapping
import math


def flatten_dict(nested_dict, parent_key='', sep='-'):
    """
    Flattens a nested dictionary

    Parameters
    ----------
    nested_dict : dict
        Nested dictionary
    parent_key : str, Optional
        Name of current parent
    sep : str, Optional. Default='-'
        Separator between the keys of different levels

    Returns
    -------
    dict
        Dictionary whose keys are flattened to a single level
    Notes
    -----
    Taken from https://stackoverflow.com/questions/6027558/flatten-nested-
    dictionaries-compressing-keys
    """
    items = []
    for key, value in nested_dict.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def print_nested_dict(nested_dict, level=0):
    """
    Prints a nested dictionary in a nested manner

    Parameters
    ----------
    nested_dict : dict
        Nested dictionary
    level : uint, internal variable. Leave unspecified
        Current depth of nested dictionary

    Returns
    -------
    None
    """
    if not isinstance(nested_dict, dict):
        raise TypeError('nested_dict should be a dict. Provided object was: {}'
                        ''.format(type(nested_dict)))
    for key, val in nested_dict.items():
        if isinstance(val, dict):
            print('\t'*level + str(key) + ' :')
            print_nested_dict(val, level=level+1)
        else:
            print('\t'*level + '{} : {}'.format(key, val))


def merge_dicts(left, right, path=None):
    """
    Merges two nested dictionaries (right into left)

    Parameters
    ----------
    left : dict
        First dictionary
    right : dict
        Second dictionary
    path : str, internal variable. Do not assign
        Internal path to current key

    Notes
    -----
    https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
    """
    if path is None: path = []
    for key in right:
        if key in left:
            if isinstance(left[key], dict) and isinstance(right[key], dict):
                merge_dicts(left[key], right[key], path + [str(key)])
            elif all([isinstance(this_dict[key], float) for this_dict in
                      [left, right]]) and math.isnan(left[key]) and math.isnan(right[key]):
                pass
            elif left[key] == right[key]:
                pass  # same leaf value
            elif isinstance(left[key], dict) and not isinstance(right[key], dict):
                merge_dicts(left[key], {'value': right[key]}, path + [str(key)])
            elif not isinstance(left[key], dict) and isinstance(right[key], dict):
                left[key] = {'value': left[key]}
                merge_dicts(left[key], right[key], path + [str(key)])
            else:
                mesg = 'Left value: {} of type: {} cannot be merged with Right: value: {} of type: {}' \
                       ''.format(left[key], type(left[key]), right[key], type(right[key]))
                raise ValueError('Conflict at %s' % '|'.join(
                    path + [str(key)]) + '. ' + mesg)
        else:
            left[key] = right[key]
    return left