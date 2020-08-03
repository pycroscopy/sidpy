# -*- coding: utf-8 -*-
"""
Utilities that assist in dictionary manipulation

Created on Thu Sep  7 21:14:25 2017

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, unicode_literals, absolute_import
from collections.abc import MutableMapping


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


