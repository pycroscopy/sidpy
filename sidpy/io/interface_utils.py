# -*- coding: utf-8 -*-
"""
Utilities for formatting strings and other input / output methods

Created on Tue Nov  3 21:14:25 2015

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import os
import sys
from warnings import warn

if sys.version_info.major == 3:
    unicode = str

__all__ = ['check_ssh', 'file_dialog']


def check_ssh():
    """
    Checks whether or not the python kernel is running locally (False) or remotely (True)

    Returns
    -------
    output : bool
        Whether or not the kernel is running over SSH (remote machine)
    """
    return 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ


def file_dialog(file_filter='H5 file (*.h5)', caption='Select File'):
    """
    Presents a File dialog used for selecting the .mat file
    and returns the absolute filepath of the selecte file\n

    Parameters
    ----------
    file_filter : String or list of strings
        file extensions to look for
    caption : (Optional) String
        Title for the file browser window

    Returns
    -------
    file_path : String
        Absolute path of the chosen file
    """
    for param in [file_filter, caption]:
        if param is not None:
            if not isinstance(param, (str, unicode)):
                raise TypeError('param must be a string')

    # Only try to use the GUI options if not over an SSH connection.
    if not check_ssh():
        try:
            from PyQt5 import QtWidgets
        except ImportError:
            warn('The required package PyQt5 could not be imported.\n',
                 'The code will check for PyQt4.')

        else:
            app = QtWidgets.QApplication([])
            path = QtWidgets.QFileDialog.getOpenFileName(caption=caption, filter=file_filter)[0]
            app.closeAllWindows()
            app.exit()
            del app

            return str(path)

        try:
            from PyQt4 import QtGui
        except ImportError:
            warn('PyQt4 also not found.  Will use standard text input.')

        else:
            app = QtGui.QApplication([])
            path = QtGui.QFileDialog.getOpenFileName(caption=caption, filter=file_filter)
            app.exit()
            del app

            return str(path)

    path = input('Enter path to datafile.  Raw Data (*.txt, *.mat, *.xls, *.xlsx) or Translated file (*.h5)')

    return str(path)


