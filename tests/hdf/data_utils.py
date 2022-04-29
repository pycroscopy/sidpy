from __future__ import division, print_function, unicode_literals, absolute_import
import os
import sys
import socket
import numpy as np
from io import StringIO
from contextlib import contextmanager
from platform import platform

sys.path.append("../../sidpy/")
from sidpy import __version__
from sidpy.base.string_utils import get_time_stamp


if sys.version_info.major == 3:
    unicode = str


def delete_existing_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def write_safe_attrs(h5_object, attrs):
    for key, val in attrs.items():
        h5_object.attrs[key] = val


def write_string_list_as_attr(h5_object, attrs):
    for key, val in attrs.items():
        h5_object.attrs[key] = np.array(val, dtype='S')


def write_aux_reg_ref(h5_dset, labels, is_spec=True):
    for index, reg_ref_name in enumerate(labels):
        if is_spec:
            reg_ref_tuple = (slice(index, index + 1), slice(None))
        else:
            reg_ref_tuple = (slice(None), slice(index, index + 1))
        h5_dset.attrs[reg_ref_name] = h5_dset.regionref[reg_ref_tuple]


def write_main_reg_refs(h5_dset, attrs):
    for reg_ref_name, reg_ref_tuple in attrs.items():
        h5_dset.attrs[reg_ref_name] = h5_dset.regionref[reg_ref_tuple]
    write_string_list_as_attr(h5_dset, {'labels': list(attrs.keys())})


@contextmanager
def capture_stdout():
    """
    context manager encapsulating a pattern for capturing stdout writes
    and restoring sys.stdout even upon exceptions

    https://stackoverflow.com/questions/17067560/intercept-pythons-print-statement-and-display-in-gui

    Examples:
    >>> with capture_stdout() as get_value:
    >>>     print("here is a print")
    >>>     captured = get_value()
    >>> print('Gotcha: ' + captured)

    >>> with capture_stdout() as get_value:
    >>>     print("here is a print")
    >>>     raise Exception('oh no!')
    >>> print('Does printing still work?')
    """
    # Redirect sys.stdout
    out = StringIO()
    sys.stdout = out
    # Yield a method clients can use to obtain the value
    try:
        yield out.getvalue
    finally:
        # Restore the normal stdout
        sys.stdout = sys.__stdout__
