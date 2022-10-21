# -*- coding: utf-8 -*-
"""
Utilities for user interfaces

Created on Tue Aug  3 21:14:25 2020

@author: Gerd Duscher, Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import os
import sys
import warnings
import numpy as np

import ipywidgets as widgets
from IPython.display import display

if sys.version_info.major == 3:
    unicode = str
    if sys.version_info.minor < 6:
        ModuleNotFoundError = ValueError


class open_file_dialog(object):
    """Widget to select directories or widgets from a list

    Works in google colab.
    The widget converts the name of the nion file to the one in Nion's swift software,
    because it is otherwise incomprehensible

    Attributes
    ----------
    dir_name: str
        name of starting directory
    extension: list of str
        extensions of files to be listed  in widget

    Methods
    -------
    get_directory
    set_options
    get_file_name

    Example
    -------
    >>from google.colab import drive
    >>drive.mount("/content/drive")
    >>file_list = pyTEMlib.file_tools.FileWidget()
    next code cell:
    >>dataset = pyTEMlib.file_tools.open_file(file_list.file_name)

    """

    def __init__(self, dir_name='.', extension=['*']):
        self.save_path = False
        self.dir_dictionary = {}
        self.dir_list = ['.', '..']
        self.display_list = ['.', '..']

        if os.path.isdir(dir_name):
            self.dir_name = dir_name
        else:
            self.dir_name = '.'
        self.get_directory(self.dir_name)
        self.dir_list = ['.']
        self.extensions = extension
        self.file_name = ''

        self.select_files = widgets.Select(
            options=self.dir_list,
            value=self.dir_list[0],
            description='Select file:',
            disabled=False,
            rows=10,
            layout=widgets.Layout(width='70%')
        )
        display(self.select_files)
        self.set_options()
        self.select_files.observe(self.get_file_name, names='value')

    def get_directory(self, directory=None):
        self.dir_name = directory
        self.dir_dictionary = {}
        self.dir_list = []
        self.dir_list = ['.', '..'] + os.listdir(directory)

    def set_options(self):
        self.dir_name = os.path.abspath(os.path.join(self.dir_name, self.dir_list[self.select_files.index]))
        dir_list = os.listdir(self.dir_name)
        file_dict = update_directory_list(self.dir_name)

        sort = np.argsort(file_dict['directory_list'])
        self.dir_list = ['.', '..']
        self.display_list = ['.', '..']
        for j in sort:
            self.display_list.append(f" * {file_dict['directory_list'][j]}")
            self.dir_list.append(file_dict['directory_list'][j])

        sort = np.argsort(file_dict['display_file_list'])

        for i, j in enumerate(sort):
            if '--' in dir_list[j]:
                self.display_list.append(f" {i:3} {file_dict['display_file_list'][j]}")
            else:
                self.display_list.append(f" {i:3}   {file_dict['display_file_list'][j]}")
            self.dir_list.append(file_dict['file_list'][j])

        self.dir_label = os.path.split(self.dir_name)[-1] + ':'
        self.select_files.options = self.display_list

    def get_file_name(self, b):
        if os.path.isdir(os.path.join(self.dir_name, self.dir_list[self.select_files.index])):
            self.set_options()

        elif os.path.isfile(os.path.join(self.dir_name, self.dir_list[self.select_files.index])):
            self.file_name = os.path.join(self.dir_name, self.dir_list[self.select_files.index])


def add_to_dict(file_dict, name):
    full_name = os.path.join(file_dict['directory'], name)
    basename, extension = os.path.splitext(name)
    size = os.path.getsize(full_name) * 2 ** -20
    display_name = name
    if len(extension) == 0:
        display_file_list = f' {name}  - {size:.1f} MB'
    elif extension[0] == 'hf5':
        if extension in ['.hf5']:
            display_file_list = f" {name}  - {size:.1f} MB"
    else:
        display_file_list = f' {name}  - {size:.1f} MB'
    file_dict[name] = {'display_string': display_file_list, 'basename': basename, 'extension': extension,
                       'size': size, 'display_name': display_name}


def update_directory_list(directory_name):
    dir_list = os.listdir(directory_name)

    file_dict = {'directory': directory_name}

    # add new files
    file_dict['file_list'] = []
    file_dict['display_file_list'] = []
    file_dict['directory_list'] = []

    for name in dir_list:
        if os.path.isfile(os.path.join(file_dict['directory'], name)):
            if name not in file_dict:
                add_to_dict(file_dict, name)
            file_dict['file_list'].append(name)
            file_dict['display_file_list'].append(file_dict[name]['display_string'])
        else:
            file_dict['directory_list'].append(name)
    return file_dict


def check_ssh():
    """
    Checks whether or not the python kernel is running locally (False) or remotely (True)

    Returns
    -------
    output : bool
        Whether or not the kernel is running over SSH (remote machine)

    Notes
    -----
    When developing workflows that need to work on remote or virtual machines
    in addition to one's own personal computer such as a laptop, this function
    is handy at letting the developer know where the code is being executed

    Examples
    --------
    >>> import sidpy
    >>> mode = sidpy.interface_utils.check_ssh()
    >>> print('Running on remote machine: {}'.format(mode))
    """
    return 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ


def get_QT_app():
    """
    Starts pyQT app if not running

    Returns: QApplication
    -------
    instance : ``QApplication.instance``
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


def openfile_dialog_QT(file_types="All files (*)", multiple_files=False,
                       file_path='.', caption="Select a file..."):
    """
    Opens a File dialog which is used in open_file() function
    This function uses pyQt5.

    Parameters
    ----------
    file_types : str, optional. Default = all
        types of files accepted
    multiple_files : bool, optional. Default = False
        Whether or not multiple files can be selected
    file_path: str, optional. Default = '.'
        path to starting or root directory
    caption: str, optional. Default = "Select a file..."
        caption of the open file dialog

    Returns
    -------
    filename : str
        full filename with absolute path and extension

    Notes
    -----
    In jupyter notebooks use ``%gui Qt`` early in the notebook.

    Examples
    --------
    >> import sidpy as sid
    >> filename = sid.io.openfile_dialog()
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
    Produces a window / dialog to allow users to specify the location and name
    of a file to save to.

    Parameters
    ----------
    initial_file : str, optional. Default = ``*.hf5``
        File extension? @gduscher to clarify
    file_path : str, optional. Default = '.'
        path to starting or root directory
    file_types :  str, optional. Default = None
        Filters for kinds of files to display in the window
    caption: str, optional. Default = "Save file as..."
        caption of the save file dialog

    Returns
    -------
    fname : str
        path to desired file

    Notes
    -----
    In jupyter notebooks use ``%gui Qt`` early in the notebook.

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
    title: str, optional. Default = 'Progress'
        Title for the progress window
    start: int, optional. Default = 0
        Start value
    stop: int, optional. Default = 100
        End value

    Returns
    -------
    progress : QtWidgets.QProgressDialog
        Progress dialog

    Examples
    --------
    >>> import sidpy
    >>> progress = sidpy.interface_utils.progress_bar('progress', 1,50)
    >>> for count in range(50):
    >>>      progress.setValue(count)
    """
    # Check whether QT is available
    warnings.warn("progress_bar() is deprecated; use tqdm package instead", warnings.DeprecationWarning)
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
