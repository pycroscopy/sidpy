"""
================================================================================
Creating and manipulating Datasets
================================================================================

**Gerd Duscher and Suhas Somnath**

08/25/2020

**This document is a simple example of how to create and manipulate Dataset
objects**

**UNDER CONSTRUCTION**
"""
# Ensure python 3 compatibility:
from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
import sys

sys.path.append('../../')
import sidpy as sid

###############################################################################
print(sid.__version__)

###############################################################################
# Creating a ``sipy.Dataset`` object
# ----------------------------------
data_set = sid.Dataset.from_array(np.zeros([4, 5, 10]), name='zeros')
print(data_set)

###############################################################################
# Note that ``data_set`` is a dask array....
# We will be improving upon the information that will be displayed when printing ``sidpy.Dataset`` objects

###############################################################################
# Accessing data within a ``Dataset``:

###############################################################################
# Slicing and dicing:

###############################################################################
# Metadata
# --------
# ``sidpy`` automatically assigns generic top-level metadata regarding the
# ``Dataset``. Users are encouraged to capture the context regarding the dataset
# by adding the quantity, units, etc. Here's how one could do that:
data_set.data_type = 'Spectral Image'
data_set.units = 'Current'
data_set.quantity = 'nA'

###############################################################################
# Scientific metadata
# ~~~~~~~~~~~~~~~~~~~
# These ``Dataset`` objects can also capture rich scientific metadata such as
# acquisition parameters, etc. as well:

###############################################################################
# Let's take a look at how we can view and access such metadata about Datasets:

###############################################################################
# Dimensions
# ----------
# The ``Dataset`` is automatically populated with generic information about
# each dimension of the ``Dataset``. It is a good idea to capture context
# regarding each of these dimensions using ``sidpy.Dimension``.
# One can provide as much or as little information about each dimension.

data_set.set_dimension(0, sid.Dimension('x', np.arange(data_set.shape[0]),
                                        units='um', quantity='Length',
                                        dimension_type='spatial'))
data_set.set_dimension(1, sid.Dimension('y', np.linspace(-2, 2, num=data_set.shape[1], endpoint=True),
                                        units='um', quantity='Length'))
data_set.set_dimension(2, sid.Dimension('bias', np.sin(np.linspace(0, 2 * np.pi, num=data_set.shape[2])),
                                        ))
###############################################################################
# One could also manually add information regarding specific components of
# dimensions associated with Datasets via:

data_set.bias.dimension_type = 'spectral'
data_set.bias.units = 'V'
data_set.bias.quantity = 'Bias'

###############################################################################
# Let's take a look at what the dataset looks like with the additional information
# regarding the dimensions

###############################################################################
# Plotting
# --------
# The ``Dataset`` object also comes with the ability to visualize its contents
# using the ``plot()`` function.

###############################################################################
# Saving
# ------
# These ``Dataset`` objects will be deleted from memory once the python script
# completes or when a notebook is closed. The information collected in a
# ``Dataset`` can reliably be stored to files using functions in sister
# packages - ``pyUSID`` and ``pyNSID`` that write the dataset according to the
# **Universal Spectroscopy and Imaging Data (USID)** or **N-dimensional
# Spectrocsopy and Imaging Data (NSID)** formats.
# Here are links to how one could save such Datasets for each package:

