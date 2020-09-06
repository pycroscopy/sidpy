# -*- coding: utf-8 -*-
"""
Abstract :class:`~sidpy.io.dataset.Dataset` base-class

Created on Tue Nov  3 15:07:16 2015

@author: Gerd Duscher

starting code from:
https://scikit-allel.readthedocs.io/en/v0.21.1/_modules/allel/model/dask.html
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
import matplotlib.pylab as plt
import string
import dask.array as da
import h5py

from .dimension import Dimension
from ..base.num_utils import get_slope
from ..base.dict_utils import print_nested_dict
from ..viz.dataset_viz import CurveVisualizer, ImageVisualizer, ImageStackVisualizer, SpectralImageVisualizer
from ..hdf.hdf_utils import is_editable_h5


def get_chunks(data, chunks=None):
    """
    Try to guess a reasonable chunk shape to use for block-wise
    algorithms operating over `data`.

    Parameters
    ----------
    data
    chunks

    Returns
    -------

    """
    if chunks is None:

        if hasattr(data, 'chunklen') and hasattr(data, 'shape'):
            # bcolz carray, chunk first dimension only
            return (data.chunklen,) + data.shape[1:]

        elif hasattr(data, 'chunks') and hasattr(data, 'shape') and \
                len(data.chunks) == len(data.shape):
            # h5py dataset
            return data.chunks

        else:
            # fall back to something simple, ~1Mb chunks of first dimension
            row = np.asarray(data[0])
            chunklen = max(1, (2**20) // row.nbytes)
            if row.shape:
                chunks = (chunklen,) + row.shape
            else:
                chunks = (chunklen,)
            return chunks

    else:

        return chunks


def ensure_array_like(data):
    """

    Parameters
    ----------
    data

    Returns
    -------

    """
    if not hasattr(data, 'shape') or not hasattr(data, 'dtype'):
        a = np.asarray(data)
        if len(a.shape) == 0:
            raise ValueError('not array-like')
        return a
    else:
        return data


def ensure_dask_array(data, chunks=None):
    """

    Parameters
    ----------
    data
    chunks

    Returns
    -------

    """
    if isinstance(data, da.Array):
        if chunks:
            data = data.rechunk(chunks)
        return data
    else:
        data = ensure_array_like(data)
        chunks = get_chunks(data, chunks)
        return da.from_array(data, chunks=chunks)


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
    -title: name of the data set
    -modality
    -source
    -axes: dictionary of Dimensions one for each data dimension
                    (the axes are dimension datasets with name, label, units, and 'dimension_type' attributes).

    -attrs: dictionary of additional metadata
    -orginal_metadata: dictionary of original metadata of file,

    -labels: returns labels of all dimensions.

    functions:
    set_dimension(axis, dimensions): set a Dimension to a specific axis
    """

    def __init__(self, *args, **kwargs):
        super(Dataset, self).__init__()

        self._units = ''
        self._title = ''
        self._data_type = ''
        self._data_descriptor = ''
        self._modality = ''
        self._source = ''

        self._h5_dataset = None
        self.metadata = {}
        self.original_metadata = {}
        self.view = None  # this will hold the figure and axis reference for a plot

    def __del__(self):
        self.hdf_close()

    def __repr__(self):
        rep = 'sipy Dataset of type {} with:\n '.format(self.data_type)
        rep = rep + super(Dataset, self).__repr__()
        rep = rep + '\n data contains: {} ({})'.format(self.quantity, self.units)
        rep = rep + '\n and Dimensions: '

        for key in self.axes:
            rep = rep + '\n  {}:  {} ({}) of size {}'.format(self.axes[key].name, self.axes[key].quantity,
                                                             self.axes[key].units, len(self.axes[key].values))
        return rep

    def hdf_close(self):
        if self.h5_dataset is not None:
            if is_editable_h5(self.h5_dataset.file):
                self.h5_dataset.file.close()
            self.h5_dataset = None

    @classmethod
    def from_array(cls, x, chunks=None, name=None, lock=False):
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

        """
        # override this as a class method to allow sub-classes to return
        # instances of themselves

        # ensure array-like
        x = ensure_array_like(x)
        if hasattr(cls, 'check_input_data'):
            cls.check_input_data(x)

        # determine chunks, guessing something reasonable if user does not
        # specify
        chunks = get_chunks(np.array(x), chunks)

        # create vanilla dask array
        darr = da.from_array(np.array(x), chunks=chunks, name=name, lock=lock)

        # view as sub-class
        cls = view_subclass(darr, cls)
        cls.data_type = 'generic'
        cls.units = 'generic'
        cls.title = 'generic'
        cls.quantity = 'generic'

        cls.modality = 'generic'
        cls.source = 'generic'
        cls.data_descriptor = 'generic'

        cls.axes = {}
        for dim in range(cls.ndim):
            # TODO: add parent to dimension to set attribute if name changes
            cls.set_dimension(dim,
                              Dimension(string.ascii_lowercase[dim],
                                        np.arange(cls.shape[dim])))
        cls.attrs = {}
        cls.group_attrs = {}
        cls.original_metadata = {}
        return cls

    def like_data(self, data,  name=None, lock=False):
        """
        Returns pysid dataset of new values but with metadata of this dataset
        - if dimension of new dataset is different from this dataset and the scale is linear,
            then this scale will be applied to the new dataset (naming and units will stay the same),
            otherwise the dimension will be generic.

        Parameters
        ----------
        data: array like
            values of new sidpy dataset
        name: optional string
            name of new sidpy dataset
        lock:  boolean

        Returns
        -------
        sidpy dataset
        """

        new_data = self.from_array(data, chunks=None, name=None, lock=False)

        new_data.data_type = self.data_type
        new_data.units = self.units
        if name is None:
            new_data.title = self.title + "_new"
        else:
            new_data.title = name
        new_data.quantity = self.quantity

        new_data.modality = self.modality
        new_data.source = self.source
        new_data.data_descriptor = ''

        for dim in range(new_data.ndim):
            # TODO: add parent to dimension to set attribute if name changes
            new_data.labels.append(string.ascii_lowercase[dim])
            if len(self.axes[dim].values) == new_data.shape[dim]:
                new_data.set_dimension(dim, self.axes[dim])
            else:
                # assuming the axis scale is equidistant
                try:
                    scale = get_slope(self.axes[dim].values)
                    axis = self.axes[dim].copy()
                    axis.values = np.arange(new_data.shape[dim])*scale
                    new_data.set_dimension(dim, axis)
                except ValueError:
                    print('using generic parameters for dimension ', dim)

        new_data.attrs = dict(self.attrs).copy()
        new_data.group_attrs = {}  # dict(self.group_attrs).copy()
        new_data.original_metadata = {}
        new_data.metadata = {}
        return new_data

    def copy(self):
        """
        actually a deep copy of this dataset.

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
            dset_copy.set_dimension(dim, self.axes[dim].copy())
        dset_copy.attrs = dict(self.attrs).copy()

        return dset_copy

    def set_dimension(self, dim, dimension):
        """
        sets the dimension for the dataset including new name and updating the axes dictionary

        Parameters
        ----------
        dim: integer - dimension of dataset
        dimension: sidpy dimension - name, values, ... of dimension

        Returns
        -------

        """
        if isinstance(dimension, Dimension):
            setattr(self, dimension.name, dimension)
            setattr(self, 'dim_{}'.format(dim), dimension)
            self.axes[dim] = dimension
        else:
            raise ValueError('dimension needs to be a sidpy dimension object')

    def view_metadata(self):
        if isinstance(self.metadata, dict):
            print_nested_dict(self.metadata)
            
    def view_original_metadata(self):
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
        if len(self.shape) == 1:
            if verbose:
                print('1D dataset')
            self.view = CurveVisualizer(self)
            plt.show()
        elif len(self.shape) == 2:
            # this can be an image or a set of line_plots
            if verbose:
                print('2D dataset')
            if self.data_type == 'image':
                self.view = ImageVisualizer(self, **kwargs)
                plt.show()
            elif self.data_type in ['spectrum_family', 'line_family', 'line_plot_family', 'spectra']:
                print('not implemented yet')
            else:
                print('not implemented yet')
        elif len(self.shape) == 3:
            if verbose:
                print('3D dataset')
            if self.data_type == 'image':
                self.view = ImageVisualizer(self, **kwargs)
                plt.show()
            elif self.data_type == 'image_map':
                pass
            elif self.data_type == 'image_stack':
                self.view = ImageStackVisualizer(self, **kwargs)
                plt.show()
            elif self.data_type == 'spectrum_image':
                self.view = SpectralImageVisualizer(self, **kwargs)
                plt.show()
            else:
                print('not implemented yet')
        else:
            print('not implemented yet')

    def get_extent(self, dimensions):
        """
        get image extend as needed i.e. in matplotlib's imshow function.
        This function works for equi or non-equi spaced axes and is suitable for subpixel accuracy of positions

        Parameters
        ----------
        dimensions: dictionary of dimensions

        Returns
        -------
        list of floats
        """
        extend = []
        for i, dim in enumerate(dimensions):
            temp = self.axes[dim].values
            start = temp[0] - (temp[1] - temp[0])/2
            end = temp[-1] + (temp[-1] - temp[-2])/2
            if i == 1:
                extend.append(end)  # y axis starts on top
                extend.append(start)
            else:
                extend.append(start)
                extend.append(end)
        return extend

    @property
    def labels(self):
        labels = []
        for key, dim in self.axes.items():
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
    def data_descriptor(self):
        return self._data_descriptor

    @data_descriptor.setter
    def data_descriptor(self, value):
        if isinstance(value, str):
            self._data_descriptor = value
        else:
            raise ValueError('data_descriptor needs to be a string')

    @property
    def data_type(self):
        return self._data_type

    @data_type.setter
    def data_type(self, value):
        if isinstance(value, str):
            self._data_type = value
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
