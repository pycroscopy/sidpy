"""
================================================================================
02. Plotting a simple image
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
import sidpy as sid

print('sidpy version: ', sid.__version__)

########################################################################################################################
# Creating a sid Dataset requires only a numpy array

image = np.zeros((8,6))

#make checker board (you can ignore how this is done exactly, the result is important)
image[::2, 1::2] = 1
image[1::2, ::2] = 1

data_set = sid.Dataset.from_array(image, name='checkerboard')
########################################################################################################################

########################################################################################################################
# Now we add additional information to the dataset
data_set.units = 'counts'
data_set.quantity = 'intensity'
data_set.data_type = 'image'
data_set.title = 'checker_board'
########################################################################################################################

########################################################################################################################
# We add the dimension axis to the Dataset, which is originally just 'generic'.
# Try it by commenting out the lines in this section.
print(np.arange(image.shape[1]))
data_set.set_dimension(0, sid.Dimension('x', np.arange(image.shape[0]), units='check', quantity='field',
                                        dimension_type='spatial'))
data_set.set_dimension(1, sid.Dimension('y', np.arange(image.shape[1]), units='check', quantity='field',
                                        dimension_type='spatial'))

########################################################################################################################

########################################################################################################################
# We plot the data
# We can set the usual key_words for plotting in the "kwargs" dictionary.
# Additionally, we have an extra keyword with name "scale_bar"; if that is set to True we get a TEM style image.
# Comment out the line with the second kwargs definition to see the difference
kwargs = {}
kwargs = {'cmap': 'viridis', 'scale_bar': True}

data_set.plot(verbose=True, **kwargs)
########################################################################################################################


