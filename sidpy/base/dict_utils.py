# -*- coding: utf-8 -*-
"""
Utilities that assist in dictionary manipulation

Created on Thu Jul  7 21:14:25 2020

@author: Suhas Somnath
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import sys

from warnings import warn
import math
if sys.version_info.major == 3:
    unicode = str
    from collections.abc import MutableMapping
else:
    from collections import MutableMapping
from sidpy.base.string_utils import validate_single_string_arg


def flatten_dict(nested_dict, separator='-'):
    """
    Flattens a nested dictionary

    Parameters
    ----------
    nested_dict : dict
        Nested dictionary
    separator : str, Optional. Default='-'
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
    if not isinstance(nested_dict, dict):
        raise TypeError('nested_dict should be a dict')
    separator = validate_single_string_arg(separator, 'separator')

    def __flatten_dict_int(nest_dict, sep, parent_key=''):
        items = []
        if sep == '_':
            repl = '-'
        else:
            repl = '_'
        for key, value in nest_dict.items():
            if not isinstance(key, str):
                key = str(key)
            if sep in key:
                key = key.replace(sep, repl)

            new_key = parent_key + sep + key if parent_key else key
            if isinstance(value, MutableMapping):
                items.extend(__flatten_dict_int(value, sep, parent_key=new_key).items())
            # nion files contain lists of dictionaries, oops
            elif isinstance(value, list):
                for i in range(len(value)):
                    if isinstance(value[i], dict):
                        for kk in value[i]:
                            items.append(('dim-' + kk + '-' + str(i), value[i][kk]))
                    else:
                        if type(value) != bytes:
                            items.append((new_key, value))
            else:
                if type(value) != bytes:
                    items.append((new_key, value))
        return dict(items)

    return __flatten_dict_int(nested_dict, separator)


def nested_dict_from_flattened_key(single_pair, separator='-'):
    """
    Converts a dictionary with a single key: value pair to a nested dictionary

    Parameters
    ----------
    single_pair : dict
        Dictionary with a single key-value pair
    separator : str, optional. Default = '-'
        Separator used to delimit the levels in the keys

    Returns
    -------
    nested_dict : dict
        Nested dictionary

    Notes
    -----
    Converts {'A|B|C': value}... to
             {'A':
                 'B': {'C': value_1,
                       }
                  }
              }
    """
    if not isinstance(single_pair, dict):
        raise TypeError('Expected dict. Provided object of type: {}'
                        ''.format(type(single_pair)))
    if len(single_pair) > 1:
        warn('This function only works on one key-value pair. Provided dict'
             'has {} pairs'.format(len(single_pair)))
    key = list(single_pair.keys())[-1]
    if not isinstance(key, (str, unicode)):
        raise TypeError('Key for provided dict not string and is instead: '
                        '{}'.format(type(key)))
    value = single_pair[key]
    # break up the key by separator
    hierarchy = key.split(separator)
    # set  actual value to the last item in the hierarchy discovered above:
    nested_dict = {hierarchy[-1]: value}
    # build the tree above by iterating in reverse, excepting the last leaf
    for parent in hierarchy[:-1][::-1]:
        nested_dict = {parent: nested_dict}
    return nested_dict


def nest_dict(flat_dict, separator='-'):
    """
    Generates a nested dictionary from a flattened dictionary

    Parameters
    ----------
    flat_dict : dict
        Dictionary whose keys are flattened to a single string with a separator
    separator : str, optional. Default = '-'
        Separator used to delimit the levels in the keys

    Returns
    -------
    nested_dict : dict
        Nested dictionary

    Notes
    -----
    flat_dict should look like {'A|B|C': V1, 'A|B|D': V2, ...}
    """
    nested_dict = dict()
    conflict_items = dict()
    for key, val in flat_dict.items():
        this_dict = nested_dict_from_flattened_key({key: val},
                                                   separator=separator)
        try:
            # merge the nested dictionaries:
            nested_dict = merge_dicts(nested_dict, this_dict)
        except ValueError as exp:
            warn(exp)
            conflict_items.update({key: val})
    if len(conflict_items) > 0:
        return nested_dict, conflict_items
    return nested_dict


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
    if path is None:
        path = []
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
