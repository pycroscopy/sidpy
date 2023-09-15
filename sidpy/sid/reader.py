# -*- coding: utf-8 -*-
"""
Abstract :class:`~sidpy.Reader` base-class

Created on Sun Aug 22 11:07:16 2020

@author: Suhas Somnath
"""


from __future__ import division, print_function, absolute_import, unicode_literals
import warnings
import abc
import sys
import os
from sidpy.base.string_utils import validate_single_string_arg, \
    validate_list_of_strings

if sys.version_info.major == 3:
    unicode = str
else:
    FileNotFoundError = ValueError


class Reader(object):
    """
    Abstract class that defines the most basic functionality of a data format
    Reader.
    A Reader extracts measurement data and metadata from binary / proprietary
    data formats into a single or set of sipy.Dataset objects
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, file_path, *args, **kwargs):
        """
        Parameters
        -----------
        file_path : str
            Path to the file that needs to be read
            
        Attributes
        ----------
        self._input_file_path : str
            Path to the file that will be read

        Notes
        -----
        * This method will check to make sure that the provided file_path is
          indeed a string and a valid file path.
        * Consider calling ``can_read()`` within ``__init__()`` for validating
          the provided file

        Raises
        ------
        FileNotFoundError
        """
        file_path = validate_single_string_arg(file_path, 'file_path')
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path + ' does not exist')
        self._input_file_path = file_path

    @abc.abstractmethod
    def can_read(self):
        warnings.warn("The 'can_read' method has been deprecated.", DeprecationWarning, stacklevel=2)
        return None
