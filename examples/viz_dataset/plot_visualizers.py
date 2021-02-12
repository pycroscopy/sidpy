"""
================================================================================
Plotting Datasets
================================================================================

**Gerd Duscher**

08/25/2020

**Please download this example and run it as a notebook by scrolling to the
bottom of this page**
"""
# Ensure python 3 compatibility:
from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
import sidpy

print(sidpy.__version__)

###############################################################################
# Plotting an Image
# -----------------
# First, we make a sidpy dataset from a numpy array
x = np.random.normal(loc=3, scale=2.5, size=(128, 128))
dset = sidpy.Dataset.from_array(x)

###############################################################################
# Next, we add some information about this dataset

dset.data_type = 'image'
dset.units = 'counts'
dset.quantity = 'intensity'

###############################################################################
# For plotting it is important to set the dimensions correctly.

dset.set_dimension(0, sidpy.Dimension('x', np.arange(dset.shape[0])*.02))
dset.x.dimension_type = 'spatial'
dset.x.units = 'nm'
dset.x.quantity = 'distance'
dset.set_dimension(1, sidpy.Dimension('y', np.arange(dset.shape[1])*.02))
dset.y.dimension_type = 'spatial'
dset.yunits = 'nm'
dset.y.quantity = 'distance'

###############################################################################
# Now we plot the dataset:
dset.plot()

###############################################################################
# Creating an Image-Stack DataSet
# -------------------------------
# In the following we will make a numpy which resembles a stack of images
# 
# In the ``sidpy Dataset`` will set the ``data_type`` to ``image_stack`` for the plotting routine to know how to plot this dataset.
# 
# The dimensions have to contain at least two ``spatial`` dimensions and one that is identifiable as a stack dimension ('stack, 'frame', 'time').
# First we make a stack of images
x = np.random.normal(loc=3, scale=2.5, size=(25, 128, 128))

dset = sidpy.Dataset.from_array(x)
dset.data_type = 'image_stack'
dset.units = 'counts'
dset.quantity = 'intensity'

dset.set_dimension(0, sidpy.Dimension('frame', np.arange(dset.shape[0])))
dset.frame.dimension_type = 'time'
dset.set_dimension(1, sidpy.Dimension('x', np.arange(dset.shape[1])*.02))
dset.x.dimension_type = 'spatial'
dset.x.units = 'nm'
dset.x.quantity = 'distance'
dset.set_dimension(2, sidpy.Dimension('y', np.arange(dset.shape[2])*.02))
dset.y.dimension_type = 'spatial'
dset.yunits = 'nm'
dset.y.quantity = 'distance'


###############################################################################
# Plotting the Dataset
# --------------------
# Please note that the scroll wheel will move you through the stack.
# 
# Zoom to an area and let it play!
# 
# Click on the ``Average`` button and then click on it again.

dset.plot()

###############################################################################
# The kwargs dictionary is used to plot the image stack in TEM style with scale bar

kwargs = {'scale_bar': True, 'cmap': 'hot'}  # or maybe 'cmap': 'gray'
 
dset.plot(verbose=True, **kwargs)

###############################################################################
# Plot Dataset as Spectral Image
# ------------------------------
# We need to change the data_type of the dataset to ``spectrum_image`` and the dimension_type of one dimension to ``spectral``.
# 
# Now the plot function plots it as a spectrum image.
# 
# Select the spectrum with the mouse (left click).

dset.data_type = 'spectrum_image'
dset.set_dimension(0, sidpy.Dimension('spectrum',np.arange(dset.shape[0])))
dset.spectrum.dimension_type = 'spectral'

dset.plot()

###############################################################################
# We make the selection more visible by setting the binning of the spectra selection.
# 
# The binning averages over the binning box.
# Run the code-cell below and look in the plot above.
# While you can make the modifications in a jupyter noteboook in a code-cell after the
# dset.plot() command is executed, that does not work in a script.
# Here we use the explicit visualization command followed by a plt.show() command.

dset.view.set_bin([20, 20])
plt.show()

###############################################################################
# The axes (and figure) instances of matplotlib can be accessed through the ``view``
# attribute of  the sidpy dataset. For example ``dset.view``.
# Again that does not work in a prgram and we use the explicit command.
# Note that you always have to keep a reference for an interactive plot (here view)

"""
<<<<<<< HEAD:examples/viz/dataset/plot_visualizers.py
view = sidpy.viz.dataset_viz.SpectralImageVisualizer(dset)
view.set_bin([40,40])
x, y = np.mgrid[0:501:100, 0:501:100] + 5
view.axes[0].scatter(x, y, color='red');
=======
###############################################################################
#
kwargs = {'scale_bar': True, 'cmap': 'hot'}
    
view = sid.viz.dataset_viz.ImageStackVisualizer(dset, **kwargs)
<<<<<<< Updated upstream:examples/viz_dataset/plot_visualizers.py
=======
>>>>>>> 608507c4c878dbbaaf7968979bd27d058695deed:examples/viz_dataset/plot_visualizers.py
>>>>>>> Stashed changes:examples/viz/dataset/plot_visualizers.py
plt.show()

###############################################################################

<<<<<<< HEAD:examples/viz/dataset/plot_visualizers.py
=======
print(dset.shape)
kwargs = {'scale_bar': True, 'cmap': 'hot'}
view = sid.dataset_viz.ImageVisualizer(dset, image_number=5, **kwargs)
<<<<<<< Updated upstream:examples/viz_dataset/plot_visualizers.py
=======
>>>>>>> 608507c4c878dbbaaf7968979bd27d058695deed:examples/viz_dataset/plot_visualizers.py
>>>>>>> Stashed changes:examples/viz/dataset/plot_visualizers.py

###############################################################################
# The generic plot command of a dispy dataset looks for the ``data_type`` to
# decide how to plot the data.
# We cn force any plot with the expliit plot command, but we need to provide the
# ``dimension_type`` as information what axis to be used for the plot.

<<<<<<< HEAD:examples/viz/dataset/plot_visualizers.py
print(dset.shape)
kwargs = {'scale_bar': True, 'cmap': 'hot'}
view = sidpy.viz.dataset_viz.ImageVisualizer(dset, image_number = 5, **kwargs)
plt.show()

###############################################################################
=======
dset.data_type = 'spectrum_image'
dset.set_dimension(0, sidpy.Dimension('spectrum',np.arange(dset.shape[0])))
dset.spectrum.dimension_type = 'spectral'
view = sidpy.viz.dataset_viz.SpectralImageVisualizer(dset)
view.set_bin([30, 40])
plt.show()

###############################################################################
#
dset.data_type = 'spectrum_image'
dset.set_dimension(0, sidpy.Dimension('spectrum',np.arange(dset.shape[0])))
dset.spectrum.dimension_type = 'spectral'
# view = SpectralImageVisualizer(dset)
# dset.plot()
"""