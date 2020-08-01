from warnings import warn
import sys
if sys.version_info.major == 3:
    unicode = str

def get_QT_app():
    ##  Starts pyQT app if not running
    try:
        from PyQt5.Qt import QApplication
    except ImportError:
        warn('The required package PyQt5 could not be imported.')

    # start qt event loop
    _instance = QApplication.instance()
    if not _instance:
        print('not_instance')
        _instance = QApplication([])
    app = _instance

    return app


def openfile_dialog(file_types="All files (*)", multiple_files=False, file_path = '.', caption="Select a file..."):
    """
    Opens a File dialog which is used in open_file() function
    This functon uses pyQt5.
    In jupyter notebooks use "%gui Qt" early in the notebook.

    Parameters
    ----------
    file_types : string of the file type filter
    file_path: string of path to directory
    caption: string of caption of the open file dialog

    Returns
    -------
    filename : full filename with absolute path and extension as a string

    Examples
    --------

    >>> import sidpy as sid
    >>>
    >>> filename = sid.io.openfile_dialog()
    >>>
    >>> print(filename)


    """
    # Check whether QT is available
    try:
        from PyQt5 import QtGui, QtWidgets, QtCore

    except ImportError:
        warn('The required package PyQt5 could not be imported.')

    ## try to find a parent the file dialog can appear on top
    try:
        app = get_QT_app()
    except:
        pass

    for param in [file_path, file_types, caption]:
        if param is not None:
            if not isinstance(param, (str, unicode)):
                raise TypeError('param must be a string')

    if len(file_path) < 2:
        path = '.'

    parent = None
    if multiple_files:
        fnames, file_filter = QtWidgets.QFileDialog.getOpenFileNames(parent, caption, file_path,
                                                                     filter=file_types,
                                                                     options=[QtCore.Qt.WindowStaysOnTopHint])
        if len(fnames) > 0:
            fname = fnames[0]
        else:
            return
    else:
        fname, file_filter = QtWidgets.QFileDialog.getOpenFileName(parent, caption, file_path,
                                                                       filter=file_types)


    if multiple_files:
        return fnames
    else:
        return str(fname)


def savefile_dialog(initial_file = '*.hf5', file_path = '.', file_types = None, caption = "Save file as ..."):
    """
        Opens a save dialog in QT and retuns an "*.hf5" file.
        In jupyter notebooks use "%gui Qt" early in the notebook.

    """
    # Check whether QT is available
    try:
        from PyQt5 import QtGui, QtWidgets, QtCore

    except ImportError:
            warn('The required package PyQt5 could not be imported.')

    else:

        for param in [file_path, initial_file, caption]:
            if param is not None:
                if not isinstance(param, (str, unicode)):
                    raise TypeError('param must be a string')

        if file_types == None:
            file_types ="All files (*)"

        try:
            app = get_QT_app()
        except:
            pass
        parent = None

        if len(file_path) < 2:
            path = '.'

        fname, file_filter = QtWidgets.QFileDialog.getSaveFileName(None, caption,
                                                               file_path + "/" + initial_file, filter=file_types)
        if len(fname) > 1:
            return fname
        else:
            return None

