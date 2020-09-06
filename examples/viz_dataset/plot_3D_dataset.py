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
# Here a 3D dataset is plotted as a image stack and as a spectral image.
# This example shows the importance of the dimensions for plotting sidpy datasets
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

data_3D = np.random.random([25, 512, 512])

data_set = sidpy.Dataset.from_array(data_3D, name='data_3D')
########################################################################################################################

########################################################################################################################
# Now we add additional information to the dataset
data_set.units = 'counts'
data_set.quantity = 'intensity'
data_set.data_type = 'image'
data_set.title = 'random 3D'

########################################################################################################################

########################################################################################################################
# We add the image dimension axis to the Dataset, which is originally just 'generic'.
# This will give enough information to plot an image
# A sidpy.Dimension needs a name and values as a minimum.
data_set.set_dimension(1, sidpy.Dimension('x', np.arange(data_set.shape[1]), units='check', quantity='field',
                                        dimension_type='spatial'))
data_set.set_dimension(2, sidpy.Dimension('y', np.arange(data_set.shape[2]), units='check', quantity='field',
                                        dimension_type='spatial'))

########################################################################################################################

########################################################################################################################
# We add the image dimension axis to the Dataset, which is originally just 'generic'.
# Here we do a longer version of the above code.
# This also illustrates how we can change the dimensional information at a later point

data_set.data_type = 'image'
data_set.set_dimension(1, sidpy.Dimension('x', np.arange(data_set.shape[1])*.02))
data_set.x.dimension_type = 'spatial'
data_set.x.units = 'nm'
data_set.x.quantity = 'distance'
data_set.set_dimension(2, sidpy.Dimension('y', np.arange(data_set.shape[2])*.02))
data_set.y.dimension_type = 'spatial'
data_set.yunits = 'nm'
data_set.y.quantity = 'distance'

data_set.plot()

########################################################################################################################

########################################################################################################################
# We add the frame dimension axis to the Dataset, which was originally just 'generic'.
# This will result in an image_stack plot if we set the dataset data_type to 'image_stack'

data_set.data_type = 'image_stack'

data_set.set_dimension(0, sidpy.Dimension('frame', np.arange(data_set.shape[0])))
data_set.frame.dimension_type = 'time'

kwargs = {'cmap': 'viridis', 'scale_bar': True}
data_set.plot(**kwargs)

########################################################################################################################

########################################################################################################################
# We change the first ``dimension_type` from 'frame' to 'spectral' and the dataset ``data_type`` to 'spectrum_image'
# This will result in an spectrum_image plot.
# To change the name of the dimension we need to set the dimension again.

data_set.data_type = 'spectrum_image'

data_set.set_dimension(0, sidpy.Dimension('spectrum',np.arange(data_set.shape[0])))
data_set.spectrum.dimension_type = 'spectral'

########################################################################################################################


########################################################################################################################
# We plot the data
# We can set the usual key_words for plotting in the "kwargs" dictionary.
# Additionally, we have an extra keyword with name "scale_bar"; if that is set to True we get a TEM style image.
# Comment out the line with the second kwargs definition to see the difference
kwargs = {}

data_set.plot(verbose=True, **kwargs)

########################################################################################################################

