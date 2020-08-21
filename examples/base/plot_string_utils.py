"""
===============================================================================
String utilities
===============================================================================

**Suhas Somnath**

8/12/2017

**This is a short walk-through of useful string utilities available in sidpy**

Introduction
--------------
Some of the functions in ``sidpy.viz.plot_utils`` fill gaps in the default matplotlib package, some were
developed for scientific applications, These functions have been developed
to substantially simplify the generation of high quality figures for journal publications.

.. tip::
    You can download and run this document as a Jupyter notebook using the link at the bottom of this page.
"""

# Ensure python 3 compatibility:
from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
from warnings import warn
import matplotlib.pyplot as plt
import subprocess
import sys


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
# Package for downloading online files:
try:
    import sidpy
except ImportError:
    warn('sidpy not found.  Will install with pip.')
    import pip
    install('sidpy')
    import sidpy

########################################################################################################################
# String formatting utilities
# ===========================
# Frequently, there is a need to print out logs on the console to inform the user about the size of files, or estimated
# time remaining for a computation to complete, etc. sidpy.string_utils has a few handy functions that help in
# formatting quantities in a human readable format.
#
# format_size()
# --------------
# One function that uses this functionality to print the size of files etc. is format_size(). While one can manually
# print the available memory in gibibytes (see above), ``format_size()`` simplifies this substantially:
mem_in_bytes = sidpy.comp_utils.get_available_memory()
print('Available memory in this machine: {}'.format(sidpy.string_utils.format_size(mem_in_bytes)))

########################################################################################################################
# format_time()
# -------------
# On the same lines, ``format_time()`` is another handy function that is great at formatting time and is often used in
# Process and Fitter to print the remaining time
print('{} seconds = {}'.format(14497.34, sidpy.string_utils.format_time(14497.34)))

########################################################################################################################
# format_quantity()
# -----------------
# You can generate your own formatting function based using the generic function: ``format_quantity()``.
# For example, if ``format_time()`` were not available, we could get the same functionality via:
units = ['msec', 'sec', 'mins', 'hours']
factors = [0.001, 1, 60, 3600]
time_value = 14497.34
print('{} seconds = {}'.format(14497.34, sidpy.string_utils.format_quantity(time_value, units, factors)))

########################################################################################################################
# formatted_str_to_number()
# -------------------------
# pyUSID also has a handy function for the inverse problem of getting a numeric value from a formatted string:
unit_names = ["MHz", "kHz"]
unit_magnitudes = [1E+6, 1E+3]
str_value = "4.32 MHz"
num_value = sidpy.string_utils.formatted_str_to_number(str_value, unit_names, unit_magnitudes, separator=' ')
print('formatted_str_to_number says: {} = {}'.format(str_value, num_value))

########################################################################################################################
# get_time_stamp()
# ----------------
# We try to use a standardized format for storing time stamps in HDF5 files. The function below generates the time
# as a string that can be easily parsed if need be
print('Current time is: {}'.format(sidpy.string_utils.get_time_stamp()))

########################################################################################################################
# clean_string_att()
# -------------------
# As mentioned in our `HDF5 and h5py primer <../beginner/plot_h5py.html>`_,
# the ``h5py`` package used for reading and manipulating HDF5 files has issues which necessitate the encoding of
# attributes whose values are lists of strings. The ``clean_string_att()`` encodes lists of
# strings correctly so that they can directly be written to HDF5 without causing any errors. All other kinds of simple
# attributes - single strings, numbers, lists of numbers are unmodified by this function.

expected = ['a', 'bc', 'def']
returned = sidpy.string_utils.clean_string_att(expected)
print('List of strings value: {} encoded to: {}'.format(expected, returned))

expected = [1, 2, 3.456]
returned = sidpy.string_utils.clean_string_att(expected)
print('List of numbers value: {} returned as is: {}'.format(expected, returned))