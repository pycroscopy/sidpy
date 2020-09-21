# -*- coding: utf-8 -*-
"""
Abstract :class:`~sidpy.io.dataset.Dataset` base-class

Created on Tue Nov  3 15:07:16 2015

@author: Gerd Duscher

starting code from:
https://scikit-allel.readthedocs.io/en/v0.21.1/_modules/allel/model/dask.html
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import sys
import numpy as np
import matplotlib.pylab as plt
import string
import dask.array as da
import h5py
from enum import Enum

from .dimension import Dimension
from ..base.num_utils import get_slope
from ..base.dict_utils import print_nested_dict
from ..viz.dataset_viz import CurveVisualizer, ImageVisualizer, ImageStackVisualizer, SpectralImageVisualizer
# from ..hdf.hdf_utils import is_editable_h5


class DataTypes(Enum):
    UNKNOWN = -1
    SPECTRUM = 1
    LINE_PLOT = 2
    LINE_PLOT_FAMILY = 3
    IMAGE = 4
    IMAGE_MAP = 5
    IMAGE_STACK = 6
    SPECTRAL_IMAGE = 7
    IMAGE_4D = 8


def view_subclass(darr, cls):
    """
    View a dask Array as an instance of a dask Array sub-class.

    Parameters
    ----------
    darr
    cls

    Returns
    -------

    """
    return cls(darr.dask, name=darr.name, chunks=darr.chunks,
               dtype=darr.dtype, shape=darr.shape)


class Dataset(da.Array):
    """
    ..autoclass::Dataset

    To instantiate from an existing array-like object,
    use :func:`Dataset.from_array` - requires numpy array, list or tuple

    This dask array is extended to have the following attributes:
    -data_type: str ('image', 'image_stack',  spectrum_image', ...
    -units: str
    -quantity: str what kind of data ('intensity', 'height', ..)
    -title: name of the data set
    -modality
    -source
    -_axes: dictionary of Dimensions one for each data dimension
                    (the axes are dimension datasets with name, label, units,
                    and 'dimension_type' attributes).

    -metadata: dictionary of additional metadata
    -original_metadata: dictionary of original metadata of file,

    -labels: returns labels of all dimensions.
    -data_descriptor: returns a label for the colorbar in matplotlib and such

    functions:
    from_array(data, name): constructs the dataset form a array like object (numpy array, dask array, ...)
    set_dimension(axis, dimensions): set a Dimension to a specific axis


    """

    def __init__(self, *args, **kwargs):
        """
        Initializes Dataset object which is essentially a Dask array
        underneath

        Attributes
        ----------
        self.quantity : str
            Physical quantity. E.g. - current
        self.units : str
            Physical units. E.g. - amperes
        self.data_type : enum
            Type of data such as Image, Spectrum, Spectral Image etc.
        self.title : str
            Title for Dataset
        self.view : Visualizer
            Instance of class appropriate for visualizing this object
        self.data_descriptor : str
            Description of this dataset
        self.modality : str
            Isn't this the same as data_type?
        self.source : str
            Source of this dataset. Such as instrument, analysis, etc.?
        self.h5_dataset : h5py.Dataset
            Reference to HDF5 Dataset object from which this Dataset was
            created
        self._axes : dict
            Dictionary of Dimension objects per dimension of the Dataset
        self.meta_data : dict
            Metadata to store relevant additional information for the dataset.
        self.original_metadata : dict
            Metadata from the original source of the dataset. This dictionary
            often contains the vendor-specific metadata or internal attributes
            of the analysis algorithm
        """
        # TODO: Consider using python package - pint for quantities
        super(Dataset, self).__init__()

        self._units = ''
        self._quantity = ''
        self._title = ''
        self._data_type = DataTypes.UNKNOWN
        self._modality = ''
        self._source = ''

        self._h5_dataset = None
        self._metadata = {}
        self._original_metadata = {}
        self._axes = {}

        self.view = None  # this will hold the figure and axis reference for a plot

    def __repr__(self):
        rep = 'sidpy Dataset of type {} with:\n '.format(self.data_type)
        rep = rep + super(Dataset, self).__repr__()
        rep = rep + '\n data contains: {} ({})'.format(self.quantity, self.units)
        rep = rep + '\n and Dimensions: '

        for key in self._axes:
            # TODO: This should be using the repr of Dimension
            rep = rep + '\n  {}:  {} ({}) of size {}'.format(self._axes[key].name, self._axes[key].quantity,
                                                             self._axes[key].units, len(self._axes[key].values))
        return rep

    def hdf_close(self):
        if self.h5_dataset is not None:
            self.h5_dataset.file.close()
            print(self.h5_dataset)

    @classmethod
    def from_array(cls, x, chunks='auto', name=None, lock=False):
        """
        Initializes a sidpy dataset from an array-like object (i.e. numpy array)
        All meta-data will be set to be generically.

        Parameters
        ----------
        x: array-like object
            the values which will populate this dataset
        chunks: optional integer or list of integers
            the shape of the chunks to be loaded
        name: optional string
            the name of this dataset
        lock: boolean

        Returns
        -------
         sidpy dataset

        """

        # create vanilla dask array
        darr = da.from_array(np.array(x), chunks=chunks, name=name, lock=lock)

        # view as sub-class
        sid_dset = view_subclass(darr, cls)
        sid_dset.data_type = 'UNKNOWN'
        sid_dset.units = 'generic'
        sid_dset.title = 'generic'
        sid_dset.quantity = 'generic'

        sid_dset.modality = 'generic'
        sid_dset.source = 'generic'

        sid_dset._axes = {}
        for dim in range(sid_dset.ndim):
            # TODO: add parent to dimension to set attribute if name changes
            sid_dset.set_dimension(dim,
                                   Dimension(string.ascii_lowercase[dim],
                                             np.arange(sid_dset.shape[dim])))
        sid_dset.metadata = {}
        sid_dset.original_metadata = {}
        return sid_dset

    def like_data(self, data, name=None, chunks='auto', lock=False):
        """
        Returns sidpy.Dataset of new values but with metadata of this dataset
        - if dimension of new dataset is different from this dataset and the scale is linear,
            then this scale will be applied to the new dataset (naming and units will stay the same),
            otherwise the dimension will be generic.

        Parameters
        ----------
        data: array like
            values of new sidpy dataset
        name: optional string
            name of new sidpy dataset
        chunks: optional list of integers
            size of chunks for dask array
        lock: optional boolean
            for dask array


        Returns
        -------
        sidpy dataset
        """

        new_data = self.from_array(data, chunks=chunks, name=name, lock=lock)

        new_data.data_type = self.data_type
        new_data.units = self.units
        if name is None:
            new_data.title = self.title + "_new"
        else:
            new_data.title = name
        new_data.quantity = self.quantity

        new_data.modality = self.modality
        new_data.source = self.source

        for dim in range(new_data.ndim):
            # TODO: add parent to dimension to set attribute if name changes
            new_data.labels.append(string.ascii_lowercase[dim])
            if len(self._axes[dim].values) == new_data.shape[dim]:
                new_data.set_dimension(dim, self._axes[dim])
            else:
                # assuming the axis scale is equidistant
                try:
                    scale = get_slope(self._axes[dim].values)
                    axis = self._axes[dim].copy()
                    axis.values = np.arange(new_data.shape[dim])*scale
                    new_data.set_dimension(dim, axis)
                except ValueError:
                    print('using generic parameters for dimension ', dim)

        new_data.metadata = dict(self.metadata).copy()
        new_data.original_metadata = {}
        return new_data

    def copy(self):
        """
        Returns a deep copy of this dataset.

        Returns
        -------
        sidpy dataset

        """
        dset_copy = Dataset.from_array(self, self.chunks, self.name)

        dset_copy.title = self.title
        dset_copy.units = self.units
        dset_copy.quantity = self.quantity
        dset_copy.data_type = self.data_type
        dset_copy.modality = self.modality
        dset_copy.source = self.source

        dset_copy.axes = {}
        for dim in range(dset_copy.ndim):
            dset_copy.set_dimension(dim, self.metadata[dim].copy())
        dset_copy.metadata = dict(self.metadata).copy()

        return dset_copy

    def rename_dimension(self, ind, name):
        """
        Renames Dimension at the specified index

        Parameters
        ----------
        ind : int
            Index of the dimension
        name : str
            New name for Dimension
        """
        if not isinstance(ind, int):
            raise ValueError('Dimension must be an integer')
        if 0 > ind >= len(self.shape):
            raise ValueError('Dimension must be an integer between 0 and {}'
                             ''.format(len(self.shape)-1))
        if not isinstance(name, str):
            raise ValueError('New Dimension name must be a string')
        delattr(self, self._axes[ind].name)
        self._axes[ind].name = name
        setattr(self, name, self._axes[ind])

    def set_dimension(self, dim, dimension):
        """
        sets the dimension for the dataset including new name and updating the axes dictionary

        Parameters
        ----------
        dim: int
            Index of dimension
        dimension: sidpy.Dimension
            Dimension object describing this dimension of the Dataset

        Returns
        -------

        """
        if isinstance(dimension, Dimension):
            setattr(self, dimension.name, dimension)
            setattr(self, 'dim_{}'.format(dim), dimension)
            self._axes[dim] = dimension
        else:
            raise ValueError('dimension needs to be a sidpy dimension object')

    def view_metadata(self):
        """
        Prints the metadata to stdout

        Returns
        -------
        None
        """
        if isinstance(self.metadata, dict):
            print_nested_dict(self.metadata)
            
    def view_original_metadata(self):
        """
        Prints the original_metadata dictionary to stdout

        Returns
        -------
        None
        """
        if isinstance(self.original_metadata, dict):
            print_nested_dict(self.original_metadata)

    def plot(self, verbose=False, **kwargs):
        """
        Plots the dataset according to the
         - shape of the sidpy Dataset,
         - data_type of the sidpy Dataset and
         - dimension_type of dimensions of sidpy Dataset
            the dimension_type 'spatial' or 'spectral' determines how a dataset is plotted.

        Recognized data_types are:
        1D: any keyword, but 'spectrum' or 'line_plot' are encouraged
        2D: 'image' or one of ['spectrum_family', 'line_family', 'line_plot_family', 'spectra']
        3D: 'image', 'image_map', 'image_stack', 'spectrum_image'
        4D: not implemented yet, but will be similar to spectrum_image.

        Parameters
        ----------
        verbose: boolean
        kwargs: dictionary for additional plotting parameters
            additional keywords (besides the matplotlib ones) for plotting are:
            - scale_bar: for images to replace axis with a scale bar inside the image

        Returns
        -------
            does not return anything but the view parameter is set with access to figure and axis.

        """

        if verbose:
            print('Shape of dataset is: ', self.shape)

        if self.data_type.value < 0:
            raise NameError('Datasets with UNKNOWN data_types cannot be plotted')
        if len(self.shape) == 1:
            if verbose:
                print('1D dataset')
            self.view = CurveVisualizer(self)
            plt.show()
        elif len(self.shape) == 2:
            # this can be an image or a set of line_plots
            if verbose:
                print('2D dataset')
            if self.data_type == DataTypes.IMAGE:
                self.view = ImageVisualizer(self, **kwargs)
                plt.show()
            elif self.data_type.value <= DataTypes['LINE_PLOT'].value:
                # self.data_type in ['spectrum_family', 'line_family', 'line_plot_family', 'spectra']:
                self.view = CurveVisualizer(self, **kwargs)
                plt.show()
            else:
                raise NotImplementedError('Datasets with data_type {} cannot be plotted, yet.'.format(self.data_type))
        elif len(self.shape) == 3:
            if verbose:
                print('3D dataset')
            if self.data_type == DataTypes.IMAGE:
                self.view = ImageVisualizer(self, **kwargs)
                plt.show()
            elif self.data_type == DataTypes.IMAGE_MAP:
                pass
            elif self.data_type == DataTypes.IMAGE_STACK:
                self.view = ImageStackVisualizer(self, **kwargs)
                plt.show()
            elif self.data_type == DataTypes.SPECTRAL_IMAGE:
                self.view = SpectralImageVisualizer(self, **kwargs)
                plt.show()
            else:
                raise NotImplementedError('Datasets with data_type {} cannot be plotted, yet.'.format(self.data_type))
        else:
            raise NotImplementedError('Datasets with data_type {} cannot be plotted, yet.'.format(self.data_type))

    def get_extent(self, dimensions):
        """
        get image extents as needed i.e. in matplotlib's imshow function.
        This function works for equi- or non-equi spaced axes and is suitable
        for subpixel accuracy of positions

        Parameters
        ----------
        dimensions: dictionary of dimensions

        Returns
        -------
        list of floats
        """
        # TODO : Should this method not be internal? i.e. start with _
        extend = []
        for ind, dim in enumerate(dimensions):
            temp = self._axes[dim].values
            start = temp[0] - (temp[1] - temp[0])/2
            end = temp[-1] + (temp[-1] - temp[-2])/2
            if ind == 1:
                extend.append(end)  # y axis starts on top
                extend.append(start)
            else:
                extend.append(start)
                extend.append(end)
        return extend

    @property
    def labels(self):
        labels = []
        for key, dim in self._axes.items():
            labels.append(dim.name)
        return labels

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        if isinstance(value, str):
            self._title = value
        else:
            raise ValueError('title needs to be a string')

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        if isinstance(value, str):
            self._units = value
        else:
            raise ValueError('units needs to be a string')

    @property
    def quantity(self):
        return self._quantity

    @quantity.setter
    def quantity(self, value):
        if isinstance(value, str):
            self._quantity = value
        else:
            raise ValueError('quantity needs to be a string')

    @property
    def data_type(self):
        return self._data_type

    @data_type.setter
    def data_type(self, value):
        if isinstance(value, str):
            if value.upper() in DataTypes._member_names_:
                self._data_type = DataTypes[value.upper()]
            else:
                self._data_type = DataTypes.UNKNOWN
                print('Supported data_types for plotting are only: ', DataTypes._member_names_)
                print('Setting data_type to UNKNOWN')
        else:
            raise ValueError('data_type needs to be a string')

    @property
    def modality(self):
        return self._modality

    @modality.setter
    def modality(self, value):
        if isinstance(value, str):
            self._modality = value
        else:
            raise ValueError('modality needs to be a string')

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, value):
        if isinstance(value, str):
            self._source = value
        else:
            raise ValueError('source needs to be a string')

    @property
    def h5_dataset(self):
        return self._h5_dataset

    @h5_dataset.setter
    def h5_dataset(self, value):
        if isinstance(value, h5py.Dataset):
            self._h5_dataset = value
        elif value is None:
            self.hdf_close()
        else:
            raise ValueError('h5_dataset needs to be a hdf5 Dataset')

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        if isinstance(value, dict):
            if sys.getsizeof(value) < 64000:
                self._metadata = value
            else:
                raise ValueError('metadata dictionary too large, please use attributes for '
                                 'large additional data sets')
        else:
            raise ValueError('metadata needs to be a python dictionary')

    @property
    def original_metadata(self):
        return self._original_metadata

    @original_metadata.setter
    def original_metadata(self, value):
        if isinstance(value, dict):
            if sys.getsizeof(value) < 64000:
                self._original_metadata = value
            else:
                raise ValueError('original_metadata dictionary too large, please use attributes for '
                                 'large additional data sets')
        else:
            raise ValueError('original_metadata needs to be a python dictionary')

    @property
    def data_descriptor(self):
        return '{} ({})'.format(self.quantity, self.units)
