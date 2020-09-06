"""
================================================================================
02. Plotting a simple spectrum
================================================================================

**Gerd Duscher**

08/25/2020

**This document is a simple example of how to plot a simple spectrum**
"""
########################################################################################################################
# Introduction
# -------------
# We just want to plot a spectrum as a1D plot with all the information in the axis of the plot.
# This example even though it is simple shows the principle of plotting a sidpy Dataset.
#
# Import all necessary packages
# -------------------------------
# For this example, we only need sidpy and numpy which comes with the standard Anaconda distribution:
#
# * ``numpy`` - for basic numerical work
# * ``h5py`` - the package that will be the focus of this primer

# Ensure python 3 compatibility:
from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np

import sys
sys.path.append('../../')
import sidpy

print('sidpy version: ', sidpy.__version__)

########################################################################################################################
# Creating a sidpy Dataset requires only a numpy array

# The different frequencies:
x_vec = np.linspace(0, 2*np.pi, 256)
# Generating the signals at the different "positions"
spectrum = np.sin(x_vec)

data_set = sidpy.Dataset.from_array(spectrum, name='signal')
########################################################################################################################

########################################################################################################################
# Now we add additional information to the dataset
data_set.units = 'a.u.'
data_set.quantity = 'intensity'
data_set.data_type = 'spectrum'
data_set.title = 'exp. spectrum'
########################################################################################################################

########################################################################################################################
# We add the dimension axis to the Dataset, which is originally just 'generic'.
# Try it by commenting out the lines in this section.
# The second command of this section shows an alternative way to manipulate the dimension information.

data_set.set_dimension(0, sidpy.Dimension('frequency', x_vec, units='Hz', quantity='frequency',
                                        dimension_type='spectral'))
data_set.frequency.units = '1/s'
########################################################################################################################

########################################################################################################################
# We plot the data

data_set.plot(verbose=True)

########################################################################################################################


