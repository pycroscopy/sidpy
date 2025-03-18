# -*- coding: utf-8 -*-
"""
Utilities for formatting strings and other input / output methods

Created on Tue Nov  3 21:14:25 2015

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import sys
from numbers import Number
from time import strftime
import numpy as np

if sys.version_info.major == 3:
    unicode = str
    from collections.abc import Iterable
else:
    from collections import Iterable


def format_quantity(value, unit_names, factors, decimals=2):
    """
    Formats the provided quantity such as time or size to appropriate strings

    Parameters
    ----------
    value : number
        value in some base units. For example - time in seconds
    unit_names : array-like
        List of names of units for each scale of the value
    factors : array-like
        List of scaling factors for each scale of the value
    decimals : uint, optional. default = 2
        Number of decimal places to which the value needs to be formatted

    Returns
    -------
    str
        String with value formatted correctly

    Examples
    --------
    >> # If ``sidpy.string_utils.format_time()`` were not available, we could
    >> # get the same functionality via:
    >> import sidpy
    >> units = ['msec', 'sec', 'mins', 'hours']
    >> factors = [0.001, 1, 60, 3600]
    >> time_value = 14497.34
    >> str_form = sidpy.string_utils.format_quantity(time_value,units,factors)
    >> print('{} seconds = {}'.format(14497.34, str_form))

    See Also
    --------
    sidpy.string_utils.format_size
    sidpy.string_utils.format_time
    """
    # assert isinstance(value, (int, float))
    if not isinstance(unit_names, Iterable):
        raise TypeError('unit_names must an Iterable')
    if not isinstance(factors, Iterable):
        raise TypeError('factors must be an Iterable')
    if len(unit_names) != len(factors):
        raise ValueError('unit_names and factors must be of the same length')
    unit_names = validate_list_of_strings(unit_names, 'unit_names')
    index = None

    for index, val in enumerate(factors):
        if value < val:
            index -= 1
            break

    index = max(0, index)  # handles sub msec

    return '{} {}'.format(np.round(value / factors[index], decimals),
                          unit_names[index])


def format_time(time_in_seconds, decimals=2):
    """
    Formats the provided time in seconds into seconds, minutes, or hours

    Parameters
    ----------
    time_in_seconds : number
        Time in seconds
    decimals : uint, optional. default = 2
        Number of decimal places to which the time needs to be formatted

    Returns
    -------
    str
        String with time formatted correctly

    Examples
    --------
    >>> import sidpy
    >>> num_secs = 14497.34
    >>> time_form = sidpy.string_utils.format_time(num_secs)
    >>> print('{} seconds = {}'.format(num_secs, time_form))
    """
    units = ['msec', 'sec', 'mins', 'hours']
    factors = [0.001, 1, 60, 3600]
    return format_quantity(time_in_seconds, units, factors, decimals=decimals)


def format_size(size_in_bytes, decimals=2):
    """
    Formats the provided size in bytes to kB, MB, GB, TB etc.

    Parameters
    ----------
    size_in_bytes : number
        size in bytes
    decimals : uint, optional. default = 2
        Number of decimal places to which the size needs to be formatted

    Returns
    -------
    str
        String with size formatted correctly

    Examples
    --------
    >>> # using the function to print available memory / RAM in system:
    >>> import sidpy
    >>> mem_in_bytes = sidpy.comp_utils.get_available_memory()
    >>> print('Available memory in this machine: {}'
    >>>       ''.format(sidpy.string_utils.format_size(mem_in_bytes)))

    See Also
    --------
    sidpy.comp_utils.get_available_memory
    """
    units = ['bytes', 'kB', 'MB', 'GB', 'TB']
    factors = 1024 ** np.arange(len(units), dtype=np.int64)
    return format_quantity(size_in_bytes, units, factors, decimals=decimals)


def formatted_str_to_number(str_val, magnitude_names, magnitude_values, separator=' '):
    """
    Takes a formatted string like '4.32 MHz' to 4.32 E+6.
    This function provides the inverse functionality of
    ``sidpy.string_utils.format_quantity``

    Parameters
    ----------
    str_val : str / unicode
        String value of the quantity. Example '4.32 MHz'
    magnitude_names : Iterable
        List of names of units like ['seconds', 'minutes', 'hours']
    magnitude_values : Iterable
        List of values (corresponding to magnitude_names) that scale the
        numeric value. Example [1, 60, 3600]
    separator : str / unicode, optional. Default = ' ' (space)
        The text that separates the numeric value and the units.

    Returns
    -------
    number
        Numeric value of the string

    Examples
    --------
    >>> import sidpy
    >>> unit_names = ["MHz", "kHz"]
    >>> unit_magnitudes = [1E+6, 1E+3]
    >>> str_value = "4.32 MHz"
    >>> num_value = sidpy.string_utils.formatted_str_to_number(str_value,
    >>>                                                       unit_names,
    >>>                                                       unit_magnitudes,
    >>>                                                       separator=' ')
    >>> print('formatted_str_to_number: {} = {}'.format(str_value, num_value))

    See Also
    --------
    sidpy.string_utils.format_quantity
    """
    [str_val] = validate_string_args(str_val, 'str_val')
    magnitude_names = validate_list_of_strings(magnitude_names,
                                               'magnitude_names')

    if not isinstance(separator, (str, unicode)):
        raise TypeError('separator must be a string')
    if not isinstance(magnitude_values, (list, tuple)):
        raise TypeError('magnitude_values must be an Iterable')
    if not np.all([isinstance(_, Number) for _ in magnitude_values]):
        raise TypeError('magnitude_values should contain numbers')
    if len(magnitude_names) != len(magnitude_values):
        raise ValueError('magnitude_names and magnitude_values should be of '
                         'the same length')

    components = str_val.split(separator)
    if len(components) != 2:
        raise ValueError('String value should be of format '
                         '"123.45<separator>Unit')

    for unit_name, scaling in zip(magnitude_names, magnitude_values):
        if unit_name == components[1]:
            # Let it raise an exception. Don't catch
            return scaling * float(components[0])


def validate_single_string_arg(value, name):
    """
    This function is to be used when validating a SINGLE string parameter for a
    function. Trims the provided value.
    Errors in the string will result in Exceptions

    Parameters
    ----------
    value : str
        Value of the parameter
    name : str
        Name of the parameter

    Returns
    -------
    str
        Cleaned string value of the parameter
    """
    if not isinstance(value, (str, unicode)):
        raise TypeError(name + ' should be a string')
    value = value.strip()
    if len(value) <= 0:
        raise ValueError(name + ' should not be an empty string')
    return value


def validate_list_of_strings(str_list, parm_name='parameter'):
    """
    This function is to be used when validating and cleaning a list of strings.
    Trims the provided strings. Errors in the strings will result in Exceptions

    Parameters
    ----------
    str_list : array-like
        list or tuple of strings
    parm_name : str, Optional. Default = 'parameter'
        Name of the parameter corresponding to this string list that will be
        reported in the raised Errors

    Returns
    -------
    array-like
        List of trimmed and validated strings when ALL objects within the list
        are found to be valid strings
    """

    if isinstance(str_list, (str, unicode)):
        return [validate_single_string_arg(str_list, parm_name)]

    if not isinstance(str_list, (list, tuple)):
        raise TypeError(parm_name + ' should be a string or list / tuple of '
                                    'strings')

    return [validate_single_string_arg(x, parm_name) for x in str_list]


def validate_string_args(arg_list, arg_names):
    """
    This function is to be used when validating string parameters for a
    function. Trims the provided strings.
    Errors in the strings will result in Exceptions

    Parameters
    ----------
    arg_list : array-like
        List of str objects that signify the value for a position argument in
        a function
    arg_names : array-like
        List of str objects with the names of the corresponding parameters in
        the function

    Returns
    -------
    array-like
        List of str objects that signify the value for a position argument in a
        function with spaces on ends removed
    """
    if isinstance(arg_list, (str, unicode)):
        arg_list = [arg_list]
    if isinstance(arg_names, (str, unicode)):
        arg_names = [arg_names]
    cleaned_args = []
    if not isinstance(arg_list, (tuple, list)):
        raise TypeError('arg_list should be a tuple or a list or a string')
    if not isinstance(arg_names, (tuple, list)):
        raise TypeError('arg_names should be a tuple or a list or a string')
    for arg, arg_name in zip(arg_list, arg_names):
        cleaned_args.append(validate_single_string_arg(arg, arg_name))
    return cleaned_args


def clean_string_att(att_val):
    """
    Replaces any unicode objects within lists with their string counterparts to
    ensure compatibility with python 3. If the attribute is indeed a list of
    unicodes, the changes will be made in-place

    Parameters
    ----------
    att_val : object
        Attribute object

    Returns
    -------
    att_val : object
        Attribute object

    Notes
    -----
    The ``h5py`` package used for reading and manipulating HDF5 files has
    issues which necessitate the encoding of attributes whose values are lists
    of strings. This method encodes lists of strings correctly so that they can
    directly be written to HDF5 without causing any errors. All other kinds of
    simple attributes - single strings, numbers, lists of numbers are
    unmodified by this function.

    Examples
    --------
    >>> import sidpy
    >>> expected = ['a', 'bc', 'def']
    >>> returned = sidpy.string_utils.clean_string_att(expected)
    >>> print('List of strings value: {} encoded to: {}'.format(expected, returned))
    >>>
    >>> expected = [1, 2, 3.456]
    >>> returned = sidpy.string_utils.clean_string_att(expected)
    >>> print('List of numbers value: {} returned as is: {}'.format(expected, returned))
    """
    try:
        if isinstance(att_val, Iterable):
            if type(att_val) in [unicode, str]:
                return att_val
            elif np.any([type(x) in [str, unicode, bytes, np.str_] for x in att_val]):
                new_atts = []
                for x in att_val:
                    if type(x)==str:
                        new_str = x.encode('utf-8')
                        new_atts.append(new_str)
                    else:
                        new_atts.append(x)

                return np.array(new_atts, dtype='S')
            elif isinstance(att_val, (list, tuple)):
                # Not sure how to do this elegantly,
                for item in att_val:
                    if not isinstance(item, (str, unicode, bytes, np.str_,
                                             Number, list)):
                        raise TypeError('Provided object was a list or tuple '
                                        'whose element was not a string or '
                                        'number but was of type: {}'
                                        ''.format(type(item)))
        if type(att_val) == np.str_:
            return str(att_val)
        return att_val
    except TypeError:
        raise TypeError('Failed to clean: {}'.format(att_val))


def get_time_stamp():
    """
    Returns the current date and time as a string formatted as:
    Year_Month_Day-Hour_Minute_Second

    Returns
    -------
    str

    Examples
    --------
    >>> import sidpy
    >>> stamp = sidpy.string_utils.get_time_stamp()
    >>> print('Current time is: {}'.format(stamp))
    """
    return strftime('%Y_%m_%d-%H_%M_%S')


def str_to_other(value):
    """
    Casts a single value encoded in a string to the appropriate python object.
    Useful when parsing numbers, boolean, etc. in text files

    Parameters
    ----------
    value : str / unicode
        String to be cast into other appropriate python object
    """
    if not isinstance(value, (str, unicode)):
        raise TypeError('Expected object of type str. Provided object was: {}'
                        ''.format(type(value)))
    if len(value.split(' ')) > 1:
        raise ValueError('Expected a string without spaces. Got: "{}"'
                         ''.format(value))
    value = value.strip()
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            return value


def remove_extra_delimiters(line, separator=' '):
    """
    Removes extra spaces (or other delimiters) between words in a line.
    Useful when parsing parameters (written by hand)

    Parameters
    ----------
    line : str / unicode
        Line to be cleaned
    separator : str / unicode, Optional. Default = ' '
        Separator between tokens

    Returns
    -------
    line : str
        Line with extra separators removed
    """
    if not isinstance(line, (str, unicode)):
        raise TypeError('line should be a string')
    if not isinstance(separator, (str, unicode)):
        raise TypeError('separator should be a string')
    if len(separator) == 0:
        raise ValueError('separator should not be empty')
    items = line.split(separator)
    real = list()
    for item in items:
        item = item.strip()
        if len(item) > 0:
            real.append(item)
    line = separator.join(real)
    return line
