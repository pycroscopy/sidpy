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
from .dataset import Dataset


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
        _, self._file_type = os.path.splitext(file_path)
        self.datasets = []

    def _list_to_dict(self):
        if isinstance(self.datasets, Dataset):
            return self.datasets
        elif isinstance(self.datasets, list):
            if all(isinstance(dataset, Dataset) for dataset in self.datasets):
                dataset_dict = {}
                for ind, dataset in enumerate(self.datasets):
                    key = f"Channel_{int(ind):03d}"
                    dataset_dict[key] = dataset
                return dataset_dict
            else:
                raise NotImplementedError('All items in the dataset list are '
                                          'expected to be sidpy.Dataset type')

        else:
            raise NotImplementedError('datasets attribute is expected to hold a '
                                      'sidpy.Dataset object or a list '
                                      'with sidpy.Dataset objects')

    @abc.abstractmethod
    def can_read(self):
        warnings.warn("The 'can_read' method has been deprecated.", DeprecationWarning, stacklevel=2)
        return None
