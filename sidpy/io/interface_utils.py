# -*- coding: utf-8 -*-
"""
Utilities for user interfaces

Created on Tue Aug  3 21:14:25 2020

@author: Gerd Duscher, Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import os
import sys

if sys.version_info.major == 3:
    unicode = str


def check_ssh():
    """
    Checks whether or not the python kernel is running locally (False) or remotely (True)

    Returns
    -------
    output : bool
        Whether or not the kernel is running over SSH (remote machine)
    """
    return 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ


def get_QT_app():
    """
    Starts pyQT app if not running

    Returns: QApplication
    -------

    """
    try:
        from PyQt5.Qt import QApplication
    except ImportError:
        raise ModuleNotFoundError('Required package PyQt5 not available')

    # start qt event loop
    _instance = QApplication.instance()
    if not _instance:
        # print('not_instance')
        _instance = QApplication([])

    return _instance


def openfile_dialog(file_types="All files (*)", multiple_files=False,
                    file_path='.', caption="Select a file..."):
    """
    Opens a File dialog which is used in open_file() function
    This functon uses pyQt5.
    In jupyter notebooks use "%gui Qt" early in the notebook.

    Parameters
    ----------
    file_types : string of the file type filter
    multiple_files : Multiple
    file_path: string of path to directory
    caption: string of caption of the open file dialog

    Returns
    -------
    filename : full filename with absolute path and extension as a string

    Examples
    --------

    >> import sidpy as sid
    >>
    >> filename = sid.io.openfile_dialog()
    >>
    >> print(filename)

    """
    # Check whether QT is available
    try:
        from PyQt5 import QtGui, QtWidgets, QtCore

    except ImportError:
        raise ModuleNotFoundError('Required package PyQt5 not available')

    # try to find a parent the file dialog can appear on top
    try:
        get_QT_app()
    except:
        pass

    for param in [file_path, file_types, caption]:
        if param is not None:
            if not isinstance(param, (str, unicode)):
                raise TypeError('param must be a string')

    parent = None
    if multiple_files:
        func = QtWidgets.QFileDialog.getOpenFileNames
        fnames, file_filter = func(parent, caption, file_path,
                                   filter=file_types,
                                   options=[QtCore.Qt.WindowStaysOnTopHint])
        if len(fnames) > 0:
            fname = fnames[0]
        else:
            return
    else:
        func = QtWidgets.QFileDialog.getOpenFileName
        fname, file_filter = func(parent, caption, file_path,
                                  filter=file_types)

    if multiple_files:
        return fnames
    else:
        return str(fname)


def savefile_dialog(initial_file='*.hf5', file_path='.',
                    file_types=None, caption="Save file as ..."):
    """
        Opens a save dialog in QT and retuns an "*.hf5" file.
        In jupyter notebooks use "%gui Qt" early in the notebook.

    """
    # Check whether QT is available
    try:
        from PyQt5 import QtGui, QtWidgets, QtCore

    except ImportError:
        raise ModuleNotFoundError('Required package PyQt5 not available')

    else:

        for param in [file_path, initial_file, caption]:
            if param is not None:
                if not isinstance(param, (str, unicode)):
                    raise TypeError('param must be a string')

        if file_types is None:
            file_types = "All files (*)"

        try:
            get_QT_app()
        except:
            pass


        func = QtWidgets.QFileDialog.getSaveFileName
        fname, file_filter = func(None, caption,
                                  file_path + "/" + initial_file,
                                  filter=file_types)
        if len(fname) > 1:
            return fname
        else:
            return None


try:
    from PyQt5 import QtWidgets

    class ProgressDialog(QtWidgets.QDialog):
        """
        Simple dialog that consists of a Progress Bar and a Button.
        Clicking on the button results in the start of a timer and
        updates the progress bar.
        """

        def __init__(self, title=''):
            super().__init__()
            self.initUI(title)

        def initUI(self, title):
            self.setWindowTitle('Progress Bar: ' + title)
            self.progress = QtWidgets.QProgressBar(self)
            self.progress.setGeometry(10, 10, 500, 50)
            self.progress.setMaximum(100)
            self.show()

        def set_value(self, count):
            self.progress.setValue(count)

except ImportError:
    pass


def progress_bar(title='Progress', start=0, stop=100):
    """
    Opens a progress bar window
    Parameters
    ----------
    title: str
    start: int
    stop: int

    Usage`
    -------
        >> progress = sid.io.progress_bar('progress', 1,50)
        >> for count in range(50):
        >>      progress.setValue(count)
    """
    # Check whether QT is available
    try:
        from PyQt5 import QtGui, QtWidgets, QtCore
    except ImportError:
        raise ModuleNotFoundError('Required package PyQt5 not available')

    try:
        get_QT_app()
    except:
        pass

    progress = QtWidgets.QProgressDialog(title, "Abort", 0, 100)
    progress.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
    progress.show()
    return progress
