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
import ipyfilechooser

if sys.version_info.major == 3:
    unicode = str
    if sys.version_info.minor < 6:
        ModuleNotFoundError = ValueError

class open_file_dialog(ipyfilechooser.FileChooser):
    def __init__(self, directory='.'):
        self._use_dir_icons = True
        super().__init__(directory) 

        
        
    def _apply_selection(self):
        super()._apply_selection()
        selected = os.path.join(
            self._selected_path,
            self._selected_filename
        )

        if os.path.isfile(selected):
            self._label.value = self._LBL_TEMPLATE.format(
                self._selected_filename,
                'blue'
            )
        else:
            self._label.value = self._LBL_TEMPLATE.format(
                self._selected_filename,
                'green'
            )

        
    def _set_form_values(self, path: str, filename: str) -> None:
        """Set the form values."""
        # Disable triggers to prevent selecting an entry in the Select
        # box from automatically triggering a new event.
        self._pathlist.unobserve(
            self._on_pathlist_select,
            names='value'
        )
        self._dircontent.unobserve(
            self._on_dircontent_select,
            names='value'
        )
        self._filename.unobserve(
            self._on_filename_change,
            names='value'
        )

        # In folder only mode zero out the filename
        if self._show_only_dirs:
            filename = ''

        # Set form values
        self._pathlist.options = ipyfilechooser.utils.get_subpaths(path)
        self._pathlist.value = path
        self._filename.value = filename

        # file/folder real names
        dircontent_real_names = ipyfilechooser.utils.get_dir_contents(
            path,
            show_hidden=self._show_hidden,
            dir_icon_append=False,
            show_only_dirs=self._show_only_dirs,
            filter_pattern=self._filter_pattern
        )

        # file/folder display names
        dircontent_display_names = ipyfilechooser.utils.get_dir_contents(
            path,
            show_hidden=self._show_hidden,
            dir_icon_append=self._use_dir_icons,
            show_only_dirs=self._show_only_dirs,
            filter_pattern=self._filter_pattern
        )
        dircontent_display_names = self.set_display_names(dircontent_real_names, dircontent_display_names)

        # Dict to map real names to display names
        self._map_name_to_disp = {
            real_name: disp_name
            for real_name, disp_name in zip(
                dircontent_real_names,
                dircontent_display_names
            )
        }

        # Dict to map display names to real names
        self._map_disp_to_name = {
            disp_name: real_name
            for real_name, disp_name in
            self._map_name_to_disp.items()
        }

        # Set _dircontent form value to display names
        self._dircontent.options = dircontent_display_names

        # If the value in the filename Text box equals a value in the
        # Select box and the entry is a file then select the entry.
        if ((filename in dircontent_real_names) and
                os.path.isfile(os.path.join(path, filename))):
            self._dircontent.value = self._map_name_to_disp[filename]
        else:
            self._dircontent.value = None

        # Reenable triggers again
        self._pathlist.observe(
            self._on_pathlist_select,
            names='value'
        )
        self._dircontent.observe(
            self._on_dircontent_select,
            names='value'
        )
        self._filename.observe(
            self._on_filename_change,
            names='value'
        )

        # Update the state of the select button
        if self._gb.layout.display is None:
            # Disable the select button if path and filename
            # - equal an existing folder in the current view
            # - equal the already selected values
            # - don't match the provided filter pattern(s)
            check1 = filename in dircontent_real_names
            check2 = os.path.isdir(os.path.join(path, filename))
            check3 = False
            check4 = False

            # Only check selected if selected is set
            if ((self._selected_path is not None) and
                    (self._selected_filename is not None)):
                selected = os.path.join(
                    self._selected_path,
                    self._selected_filename
                )
                check3 = os.path.join(path, filename) == selected

            # Ensure only allowed extensions are used
            if self._filter_pattern:
                check4 = not ipyfilechooser.utils.match_item(filename, self._filter_pattern)

            if (check1 and check2) or check3 or check4:
                self._select.disabled = True
            else:
                self._select.disabled = False
        
    
    def set_display_names(self, dircontent_real_names, dircontent_display_names):
        
        for i in range(len(dircontent_display_names)):
            name = dircontent_display_names[i]
            full_name = os.path.join(self._pathlist.value, dircontent_real_names[i])
            if os.path.isfile(full_name):
                size = os.path.getsize(full_name) * 2 ** -20
                basename, extension = os.path.splitext(name)
                if extension in ['.hf5']:
                    dircontent_display_names[i] = f" {dircontent_display_names[i]:50}  -- {size:.1f} MB"
                else:
                    dircontent_display_names[i] = dircontent_display_names[i]
            
        return dircontent_display_names

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
