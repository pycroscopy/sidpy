# -*- coding: utf-8 -*-
"""
Abstract :class:`~sidpy.Reader` base-class

Created on Sun Aug 22 11:07:16 2020

@author: Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals
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
    def read(self, *args, **kwargs):
        """
        Extracts the data and metadata from the provided file and embeds this
        information in one or more ``sidpy.Dataset`` objects that are returned
        from this method

        Returns
        -------
        objs : ``sidpy.Dataset`` or list of ``sidpy.Dataset`` objects

        Raises
        ------
        NotImplementedError : if the child class does not implement this method

        Notes
        -----
        Do **not** accept the file path at ``read``. Use self._input_file_path
        when implementing this method
        """
        raise NotImplementedError('The read method needs to be '
                                  'implemented by the child class')

    def can_read(self, *args, **kwargs):
        """
        Checks whether the provided file can be read by this reader.

        This basic function compares the file extension against the
        ``extension`` keyword argument. If the extension matches, this function
        returns True

        Parameters
        ----------
        extension : str or iterable of str, Optional. Default = None
            File extension for the input file.

        Returns
        -------
        file_path : str
            Path to the file that needs to be provided to read()
            if the provided file was indeed a valid file
            Else, None

        Raises
        ------
        NotImplementedError : if this function is called for this or a child
        class that does not provide the ``extension`` keyword argument

        Notes
        -----
        It is recommended to add additional checks as necessary to ensure that
        the translator can indeed read the given file such as by validating the
        headers or similar metadata.
        """
        targ_ext = kwargs.get('extension', None)
        if not targ_ext:
            raise NotImplementedError('Either can_read() has not been '
                                      'implemented by this Reader or the '
                                      '"extension" keyword argument was '
                                      'missing')
        if isinstance(targ_ext, (str, unicode)):
            targ_ext = [targ_ext]
        targ_ext = validate_list_of_strings(targ_ext,
                                            parm_name='(keyword argument) '
                                                      '"extension"')

        # Get rid of any '.' separators that may be in the list of extensions
        # Also turn to lower case for case insensitive comparisons
        targ_ext = [item.replace('.', '').lower() for item in targ_ext]

        file_path = os.path.abspath(self._input_file_path)
        extension = os.path.splitext(file_path)[1][1:]

        # Ensure extension is lower case just like targets above
        extension = extension.lower()

        if extension in targ_ext:
            return file_path
        else:
            return None
