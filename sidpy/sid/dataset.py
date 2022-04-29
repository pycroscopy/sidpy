# -*- coding: utf-8 -*-
"""
Abstract :class:`~sidpy.io.dataset.Dataset` base-class

Created on Tue Nov  3 15:07:16 2015

@author: Gerd Duscher

Modified by Mani Valleti.

Look up dask source code to understand how numerical functions are implemented

starting code from:
https://scikit-allel.readthedocs.io/en/v0.21.1/_modules/allel/model/dask.html
"""

from __future__ import division, print_function, absolute_import, unicode_literals
from hashlib import new
from functools import wraps
from re import A
import sys
from collections.abc import Iterable, Iterator, Mapping
import warnings

import dask.array.core
import numpy as np
import matplotlib.pylab as plt
import string
import dask.array as da

from enum import Enum

from .dimension import Dimension, DimensionType
from ..base.num_utils import get_slope
from ..base.dict_utils import print_nested_dict
from ..viz.dataset_viz import CurveVisualizer, ImageVisualizer, ImageStackVisualizer
from ..viz.dataset_viz import SpectralImageVisualizer, FourDimImageVisualizer
# from ..hdf.hdf_utils import is_editable_h5


class DataType(Enum):
    UNKNOWN = -1
    SPECTRUM = 1
    LINE_PLOT = 2
    LINE_PLOT_FAMILY = 3
    IMAGE = 4
    IMAGE_MAP = 5
    IMAGE_STACK = 6
    SPECTRAL_IMAGE = 7
    IMAGE_4D = 8


def view_subclass(dask_array, cls):
    """
    View a dask Array as an instance of a dask Array subclass.

    Parameters
    ----------
    dask_array
    cls

    Returns
    -------
    cls: sidpy.Dataset
    """
    
    return cls(dask_array.dask, name=dask_array.name, chunks=dask_array.chunks,
               dtype=dask_array.dtype, shape=dask_array.shape)


class Dataset(da.Array):
    """
    ..autoclass::Dataset

    To instantiate from an existing array-like object,
    use :func:`Dataset.from_array` - requires numpy array, list or tuple

    This dask array is extended to have the following attributes:
    -data_type: DataTypes ('image', 'image_stack',  spectral_image', ...
    -units: str
    -quantity: str what kind of data ('intensity', 'height', ..)
    -title: title of the data set
    -modality: character of data such as 'STM, 'AFM', 'TEM', 'SEM', 'DFT', 'simulation', ..)
    -source: origin of data such as acquisition instrument ('Nion US100', 'VASP', ..)
    -_axes: dictionary of Dimensions one for each data dimension
                    (the axes are dimension datasets with name, label, units,
                    and 'dimension_type' attributes).

    -metadata: dictionary of additional metadata
    -original_metadata: dictionary of original metadata of file,

    -labels: returns labels of all dimensions.
    -data_descriptor: returns a label for the colorbar in matplotlib and such

    functions:
    -from_array(data, title): constructs the dataset form an array like object (numpy array, dask array, ...)
    -like_data(data,title): constructs the dataset form an array like object and copies attributes and
    metadata from parent dataset
    -copy()
    -plot(): plots dataset dependent on data_type and dimension_types.
    -get_extent(): extent to be used with imshow function of matplotlib
    -set_dimension(axis, dimensions): set a Dimension to a specific axis
    -rename_dimension(dimension, name): renames attribute of dimension
    -view_metadata: pretty plot of metadata dictionary
    -view_original_metadata: pretty plot of original_metadata dictionary
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
            character of data such as 'STM', 'TEM', 'DFT'
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
        self._data_type = DataType.UNKNOWN
        self._modality = ''
        self._source = ''

        self._h5_dataset = None
        self._metadata = {}
        self._original_metadata = {}
        self._axes = {}

        self.view = None  # this will hold the figure and axis reference for a plot

    def __repr__(self):
        rep = 'sidpy.Dataset of type {} with:\n '.format(self.data_type.name)
        rep = rep + super(Dataset, self).__repr__()
        rep = rep + '\n data contains: {} ({})'.format(self.quantity, self.units)
        rep = rep + '\n and Dimensions: '

        for key in self._axes:
            rep = rep + '\n'+self._axes[key].__repr__()

        if hasattr(self, 'metadata'):
            if len(self.metadata) > 0:
                rep = rep + '\n with metadata: {}'.format(list(self.metadata.keys()))
        return rep

    def hdf_close(self):
        if self.h5_dataset is not None:
            self.h5_dataset.file.close()
            print(self.h5_dataset)

    @classmethod
    def from_array(cls, x, title='generic', chunks='auto',  lock=False,
                   datatype='UNKNOWN', units='generic', quantity='generic',
                   modality='generic', source='generic', **kwargs):
        """
        Initializes a sidpy dataset from an array-like object (i.e. numpy array)
        All meta-data will be set to be generically.

        Parameters
        ----------
        x: array-like object
            the values which will populate this dataset
        chunks: optional integer or list of integers
            the shape of the chunks to be loaded
        title: optional string
            the title of this dataset
        lock: boolean
        datatype: str or sidpy.DataType
            data type of set: i.e.: 'image', spectrum', ..
        units: str
            units of dataset i.e. counts, A
        quantity: str
            quantity of dataset like intensity
        modality: str
            describes what kind of data
        source: str
            provenance of data
        kwargs: dict
            dictionary of additional information
        Returns
        -------
        sidpy dataset

        """

        # create vanilla dask array
        if isinstance(x, da.Array):
            dask_array = x
        else:
            dask_array = da.from_array(np.array(x), chunks=chunks, lock=lock)

        # view as subclass
        sid_dataset = view_subclass(dask_array, cls)
        sid_dataset.data_type = datatype
        sid_dataset.units = units
        sid_dataset.title = title
        sid_dataset.quantity = quantity

        sid_dataset.modality = modality
        sid_dataset.source = source

        sid_dataset._axes = {}
        for dim in range(sid_dataset.ndim):
            # TODO: add parent to dimension to set attribute if name changes
            sid_dataset.set_dimension(dim,
                                      Dimension(np.arange(sid_dataset.shape[dim]), string.ascii_lowercase[dim]))
        sid_dataset.metadata = {}
        sid_dataset.original_metadata = {}
        return sid_dataset

    def like_data(self, data, title=None, chunks='auto', lock=False, **kwargs):
        """
        Returns sidpy.Dataset of new values but with metadata of this dataset
        - if dimension of new dataset is different from this dataset and the scale is linear,
            then this scale will be applied to the new dataset (naming and units will stay the same),
            otherwise the dimension will be generic.
        -Additional functionality to override numeric functions
        Parameters
        ----------
        data: array like
            values of new sidpy dataset
        title: optional string
            title of new sidpy dataset
        chunks: optional list of integers
            size of chunks for dask array
        lock: optional boolean
            for dask array


        Returns
        -------
        sidpy dataset
        """
        title_suffix = kwargs.get('title_suffix', '')
        title_prefix = kwargs.get('title_prefix', '')
        reset_quantity = kwargs.get('reset_quantity', False)
        reset_units = kwargs.get('reset_units', False)
        checkdims = kwargs.get('checkdims', True)

        new_data = self.from_array(data, chunks=chunks, lock=lock)
        
        new_data.data_type = self.data_type
        
        # units
        if reset_units:
            new_data.units = 'generic'
        else:
            new_data.units = self.units

        if title is not None:
            new_data.title = title
        else:
            if title_prefix and title_suffix:
                new_data.title = self.title
            else:
                new_data.title = self.title + '_new'
        
        new_data.title = title_prefix+new_data.title+title_suffix
        
        # quantity
        if reset_quantity:
            new_data.quantity = 'generic'
        else:
            new_data.quantity = self.quantity

        new_data.modality = self.modality
        new_data.source = self.source

        if checkdims:
            for dim in range(new_data.ndim):
                # TODO: add parent to dimension to set attribute if name changes
                if len(self._axes[dim].values) == new_data.shape[dim]:
                    new_data.set_dimension(dim, self._axes[dim])
                else:
                    # assuming the axis scale is equidistant
                    try:
                        scale = get_slope(self._axes[dim])
                        # axis = self._axes[dim].copy()
                        axis = Dimension(np.arange(new_data.shape[dim])*scale, self._axes[dim].name)
                        axis.quantity = self._axes[dim].quantity
                        axis.units = self._axes[dim].units
                        axis.dimension_type = self._axes[dim].dimension_type

                        new_data.set_dimension(dim, axis)

                    except ValueError:
                        print('using generic parameters for dimension ', dim)

        new_data.metadata = dict(self.metadata).copy()
        new_data.original_metadata = {}
        return new_data

    def __reduce_dimensions(self, new_dataset, axes, keepdims=False):
        new_dataset._axes = {}
        if not keepdims:
            i = 0
            for key, dim in self._axes.items():
                new_dim = dim.copy()
                if key not in axes:
                    new_dataset.set_dimension(i, new_dim)
                    i += 1
        
        if keepdims:
            for key, dim in self._axes.items():
                new_dim = dim.copy()
                if key in axes:
                    new_dim = Dimension(np.arange(1), name=new_dim.name,
                                        quantity=new_dim.quantity, units=new_dim.units,
                                        dimension_type=new_dim.dimension_type)
                new_dataset.set_dimension(key, new_dim)
        
        return new_dataset

    def __rearrange_axes(self, new_dataset, new_order=None):
        """Rearranges the dimension order of the current instance
        Parameters:
            new_order: list or tuple of integers

        All the dimensions that are not in the new_order are deleted
        """
        new_dataset._axes = {}

        for i, dim in enumerate(new_order):
            new_dataset.set_dimension(i, self._axes[dim])
        
        return new_dataset        

    def copy(self):
        """
        Returns a deep copy of this dataset.

        Returns
        -------
        sidpy dataset

        """
        dataset_copy = Dataset.from_array(self, self.title, self.chunks)

        dataset_copy.title = self.title
        dataset_copy.units = self.units
        dataset_copy.quantity = self.quantity
        dataset_copy.data_type = self.data_type
        dataset_copy.modality = self.modality
        dataset_copy.source = self.source

        dataset_copy._axes = {}
        for dim in self._axes:
            dataset_copy.set_dimension(dim, self._axes[dim])
        dataset_copy.metadata = dict(self.metadata).copy()

        return dataset_copy

    def __validate_dim(self, ind, name):
        """
        Validates the provided index for a Dimension object

        Parameters
        ----------
        ind : int
            Index of the dimension

        Raises
        -------
        TypeError : if ind is not an integer
        IndexError : if ind is less than 0 or greater than maximum allowed
            index for Dimension
        """
        if not isinstance(ind, int):
            raise TypeError('Dimension must be an integer')
        if (0 > ind) or (ind >= self.ndim):
            raise IndexError('Dimension must be an integer between 0 and {}'
                             ''.format(self.ndim-1))
        for key, dim in self._axes.items():
            if key != ind:
                if name == dim.name:
                    raise ValueError('name: {} already used, but must be unique'.format(name))

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
        self.__validate_dim(ind, name)
        if not isinstance(name, str):
            raise TypeError('New Dimension name must be a string')
        delattr(self, self._axes[ind].name)
        self._axes[ind].name = name
        setattr(self, name, self._axes[ind])

    def set_dimension(self, ind, dimension):
        """
        sets the dimension for the dataset including new name and updating the axes dictionary

        Parameters
        ----------
        ind: int
            Index of dimension
        dimension: sidpy.Dimension
            Dimension object describing this dimension of the Dataset

        Returns
        -------

        """
        if not isinstance(dimension, Dimension):
            raise TypeError('dimension needs to be a sidpy.Dimension object')
        self.__validate_dim(ind, dimension.name)
        # delattr(self, self._axes[ind].name)
        setattr(self, dimension.name, dimension)
        setattr(self, 'dim_{}'.format(ind), dimension)
        self._axes[ind] = dimension

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

    def plot(self, verbose=False, figure=None, **kwargs):
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
        figure: matplotlib figure object
            figure reference to plo on

        Returns
        -------
        self.view.fig: matplotlib figure reference

        """

        if verbose:
            print('Shape of dataset is: ', self.shape)

        if self.data_type.value < 0:
            raise NameError('Datasets with UNKNOWN data_types cannot be plotted')
        
        if len(self.shape) == 1:
            if verbose:
                print('1D dataset')
            self.view = CurveVisualizer(self, figure=figure, **kwargs)
            # plt.show()
        elif len(self.shape) == 2:
            # this can be an image or a set of line_plots
            if verbose:
                print('2D dataset')
            if self.data_type == DataType.IMAGE:
                self.view = ImageVisualizer(self, figure=figure, **kwargs)
                # plt.show()
            elif self.data_type.value <= DataType['LINE_PLOT'].value:
                # self.data_type in ['spectrum_family', 'line_family', 'line_plot_family', 'spectra']:
                self.view = CurveVisualizer(self, figure=figure, **kwargs)
                # plt.show()
            else:
                raise NotImplementedError('Datasets with data_type {} cannot be plotted, yet.'.format(self.data_type))
        elif len(self.shape) == 3:
            if verbose:
                print('3D dataset')
            if self.data_type == DataType.IMAGE:
                self.view = ImageVisualizer(self, figure=figure, **kwargs)
                # plt.show()
            elif self.data_type == DataType.IMAGE_MAP:
                pass
            elif self.data_type == DataType.IMAGE_STACK:
                self.view = ImageStackVisualizer(self, figure=figure, **kwargs)
                # plt.show()
            elif self.data_type == DataType.SPECTRAL_IMAGE:
                self.view = SpectralImageVisualizer(self, figure=figure, **kwargs)
                # plt.show()
            else:
                raise NotImplementedError('Datasets with data_type {} cannot be plotted, yet.'.format(self.data_type))
        elif len(self.shape) == 4:
            if verbose:
                print('4D dataset')
            if self.data_type == DataType.IMAGE:
                self.view = ImageVisualizer(self, **kwargs)
                plt.show()
            elif self.data_type == DataType.IMAGE_MAP:
                pass
            elif self.data_type == DataType.IMAGE_STACK:
                self.view = ImageStackVisualizer(self, figure=figure, **kwargs)
                plt.show()
            elif self.data_type == DataType.SPECTRAL_IMAGE:
                self.view = SpectralImageVisualizer(self, figure=figure, **kwargs)
                plt.show()
            elif self.data_type == DataType.IMAGE_4D:
                self.view = FourDimImageVisualizer(self, figure=figure, **kwargs)
                plt.show()
                if verbose:
                    print('4D dataset')
            else:
                raise NotImplementedError('Datasets with data_type {} cannot be plotted, yet.'.format(self.data_type))
        else:
            raise NotImplementedError('Datasets with data_type {} cannot be plotted, yet.'.format(self.data_type))
        return self.view.fig

    def get_thumbnail(self, figure=None, thumbnail_size=128):
        """
        Creates a thumbnail which is stored in the thumbnail attibute of sidpy Datset


        Parameters
        ----------
        thumbnail_size: int
            size of icon in pixels (length of square)
            
        Returns
        -------
        thumbnail: numpy.ndarray
        """
        
        import imageio
        # Thumbnail configurations for matplotlib
        kwargs = {'figsize': (1, 1), 'colorbar': False, 'set_title': False}
        view = self.plot(figure=figure, **kwargs)
        for axis in view.axes:
            axis.set_axis_off()
        
        # Creating Thumbnail as png image
        view.savefig('thumb.png', dpi=thumbnail_size)
        self.thumbnail = imageio.imread('thumb.png')

        return self.thumbnail

    def get_extent(self, dimensions):
        """
        get image extents as needed i.e. in matplotlib's imshow function.
        This function works for equi- or non-equi spaced axes and is suitable
        for subpixel accuracy of positions

        Parameters
        ----------
        dimensions: list of dimensions

        Returns
        -------
        list of floats
        """
        extent = []
        for ind, dim in enumerate(dimensions):
            temp = self._axes[dim].values
            start = temp[0] - (temp[1] - temp[0])/2
            end = temp[-1] + (temp[-1] - temp[-2])/2
            if ind == 1:
                extent.append(end)  # y-axis starts on top
                extent.append(start)
            else:
                extent.append(start)
                extent.append(end)
        return extent

    def get_dimensions_by_type(self, dims_in):
        """ get dimension by dimension_type name

        Parameters
        ----------
        dims_in: dimension_type/str or list of dimension_types/string


        Returns
        -------
        dims_out: list of [index, dimension]
            the kind of dimensions specified in input in numerical order of the dataset, not the input!
        """

        if isinstance(dims_in, (str, DimensionType)):
            dims_in = [dims_in]
        for i in range(len(dims_in)):
            if isinstance(dims_in[i], str):
                dims_in[i] = DimensionType[dims_in[i].upper()]
        dims_out = []
        for dim, axis in self._axes.items():
            if axis.dimension_type in dims_in:
                dims_out.append(dim)  # , self._axes[dim]])
        return dims_out

    def get_image_dims(self):
        """Get all spatial dimensions"""

        image_dims = []
        for dim, axis in self._axes.items():
            if axis.dimension_type == DimensionType.SPATIAL:
                image_dims.append(dim)
        return image_dims

    def get_spectrum_dims(self):
        """Get all spectral dimensions"""

        image_dims = []
        for dim, axis in self._axes.items():
            if axis.dimension_type == DimensionType.SPECTRAL:
                image_dims.append(dim)
        return image_dims

    @property
    def labels(self):
        labels = []
        for key, dim in self._axes.items():
            labels.append('{} ({})'.format(dim.quantity, dim.units))
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
            if value.upper() in DataType._member_names_:
                self._data_type = DataType[value.upper()]
            else:
                self._data_type = DataType.UNKNOWN
                raise Warning('Supported data_types for plotting are only: ', DataType._member_names_)

        elif isinstance(value, DataType):
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
        self._h5_dataset = value

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

    def fft(self, dimension_type=None):
        """ Gets the FFT of a sidpy.Dataset of any size

        The data_type of the sidpy.Dataset determines the dimension_type over which the
        fourier transform is performed over, if the dimension_type is not set explicitly.

        The fourier transformed dataset is automatically shifted to center of dataset.

        Parameters
        ----------
        dimension_type: None, str, or sidpy.DimensionType - optional
            dimension_type over which fourier transform is performed, if None an educated guess will determine
            that from dimensions of sidpy.Dataset

        Returns
        -------
        fft_dset: 2D or 3D complex sidpy.Dataset (not tested for higher dimensions)
            2 or 3 dimensional matrix arranged in the same way as input

        Example
        -------
        >> fft_dataset = sidpy_dataset.fft()
        >> fft_dataset.plot()
        """

        if dimension_type is None:
            # test for data_type of sidpy.Dataset
            if self.data_type.name in ['IMAGE_MAP', 'IMAGE_STACK', 'SPECTRAL_IMAGE', 'IMAGE_4D']:
                dimension_type = self.dim_2.dimension_type
            else:
                dimension_type = self.dim_0.dimension_type

        if isinstance(dimension_type, str):
            dimension_type = DimensionType[dimension_type.upper()]

        if not isinstance(dimension_type, DimensionType):
            raise TypeError('Could not identify a dimension_type to perform Fourier transform on')

        axes = self.get_dimensions_by_type(dimension_type)
        if dimension_type.name in ['SPATIAL', 'RECIPROCAL']:
            if len(axes) != 2:
                raise TypeError('sidpy dataset of type', self.data_type,
                                ' has no obvious dimension over which to perform fourier transform, please specify')
            if dimension_type.name == 'SPATIAL':
                new_dimension_type = DimensionType.RECIPROCAL
            else:
                new_dimension_type = DimensionType.SPATIAL

        elif dimension_type.name == 'SPECTRAL':
            if len(axes) != 1:
                raise TypeError('sidpy dataset of type', self.data_type,
                                ' has no obvious dimension over which to perform fourier transform, please specify')
            new_dimension_type = DimensionType.SPECTRAL
        else:
            raise NotImplementedError('fourier transform not implemented for dimension_type ', dimension_type.name)

        fft_transform = np.fft.fftshift(da.fft.fftn(self, axes=axes))
        fft_dset = self.like_data(fft_transform)
        fft_dset.units = 'a.u.'
        fft_dset.modality = 'fft'

        units_x = '1/' + self._axes[axes[0]].units
        fft_dset.set_dimension(axes[0],
                               Dimension(np.fft.fftshift(np.fft.fftfreq(self.shape[axes[0]],
                                                                        d=get_slope(self._axes[axes[0]].values))),
                                         name='u', units=units_x, dimension_type=new_dimension_type,
                                         quantity='reciprocal'))
        if len(axes) > 1:
            units_y = '1/' + self._axes[axes[1]].units
            fft_dset.set_dimension(axes[1],
                                   Dimension(np.fft.fftshift(np.fft.fftfreq(self.shape[axes[1]],
                                                                            d=get_slope(self._axes[axes[1]].values))),
                                             name='v', units=units_y, dimension_type=new_dimension_type,
                                             quantity='reciprocal_length'))
        return fft_dset

    # #####################################################
    # Original dask.array functions replaced
    # ##################################################

    def __eq__(self, other):  # TODO: Test __eq__
        if not isinstance(other, Dataset):
            return False
        # if (self.__array__() == other.__array__()).all():
        if (self.__array__().__eq__(other.__array__())).all():
            if self._units != other._units:
                return False
            if self._quantity != other._quantity:
                return False
            if self._source != other._source:
                return False
            if self._data_type != other._data_type:
                return False
            if self._modality != other._modality:
                return False
            if self._axes != other._axes:
                return False
            return True
        return False

    @property
    def T(self):
        return self.transpose()

    def abs(self):
        return self.like_data(super().__abs__(), title_suffix='_absolute_value')

    ######################################################
    # Original dask.array functions handed through
    ##################################################
    @property
    def real(self):
        return self.like_data(super().real)

    @property
    def imag(self):
        return self.like_data(super().imag)

    def reduce_dims(original_method):
        @wraps(original_method)
        def wrapper_method(self, *args, **kwargs):
            result, arguments = original_method(self, *args, **kwargs)
            axis, keepdims = arguments.get('axis'), arguments.get('keepdims', False)
            if axis is None and not keepdims:
                return result
            if axis is None:
                axes = list(np.arange(self.ndim))
            elif isinstance(axis, int):
                axes = [axis]
            else:
                axes = list(axis)
            
            return self.__reduce_dimensions(result, axes, keepdims)
        
        return wrapper_method

    @reduce_dims
    def all(self, axis=None, keepdims=False, split_every=None, out=None):
        if axis is None and not keepdims:
            result = bool(super().all())
        
        else:
            result = self.like_data(super().all(axis=axis, keepdims=keepdims,
                                                split_every=split_every, out=out), checkdims=False)
        return result, locals().copy()

    @reduce_dims
    def any(self, axis=None, keepdims=False, split_every=None, out=None):
        if axis is None and not keepdims:
            result = bool(super().any())
        
        else:
            result = self.like_data(super().any(axis=axis, keepdims=keepdims,
                                                split_every=split_every, out=out), checkdims=False)
        return result, locals().copy()

    @reduce_dims
    def min(self, axis=None, keepdims=False, split_every=None, out=None):
        if axis is None and not keepdims:
            result = float(super().min())
        else:
            result = self.like_data(super().min(axis=axis, keepdims=keepdims,
                                                split_every=split_every, out=out), checkdims=False)
        return result, locals().copy()
    
    @reduce_dims
    def max(self, axis=None, keepdims=False, split_every=None, out=None):
        if axis is None and not keepdims:
            result = float(super().max())
        else:
            result = self.like_data(super().max(axis=axis, keepdims=keepdims,
                                                split_every=split_every, out=out), checkdims=False)
        return result, locals().copy()

    @reduce_dims
    def sum(self, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
        if axis is None and not keepdims:
            result = float(super().sum())
        
        else:
            result = self.like_data(super().sum(axis=axis, dtype=dtype, keepdims=keepdims,
                                                split_every=split_every, out=out), checkdims=False)
        return result, locals().copy()
    
    @reduce_dims
    def mean(self, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
        if axis is None and not keepdims:
            result = float(super().mean())
        else:
            result = self.like_data(super().mean(axis=axis, dtype=dtype, keepdims=keepdims,
                                                 split_every=split_every, out=out), checkdims=False)
        return result, locals().copy()
    
    @reduce_dims
    def std(self, axis=None, dtype=None, keepdims=False, ddof=0, split_every=None, out=None):
        if axis is None and not keepdims:
            result = float(super().std())
        else:
            result = self.like_data(super().std(axis=axis, dtype=dtype, keepdims=keepdims,
                                                ddof=0, split_every=split_every, out=out), checkdims=False)
        return result, locals().copy()

    @reduce_dims
    def var(self, axis=None, dtype=None, keepdims=False, ddof=0, split_every=None, out=None):
        if axis is None and not keepdims:
            result = float(super().var())
        else:
            result = self.like_data(super().var(axis=axis, dtype=dtype, keepdims=keepdims,
                                                ddof=0, split_every=split_every, out=out), checkdims=False)
        return result, locals().copy()
    
    @reduce_dims
    def argmin(self, axis=None, split_every=None, out=None):
        if axis is None:
            result = int(super().argmin(axis=axis, split_every=split_every, out=out))
        else:
            result = self.like_data(super().argmin(axis=axis, split_every=split_every, out=out),
                                    title_suffix='_argmin_indices', reset_units=True, reset_quantity=True,
                                    check_dims=False)
        return result, locals().copy()
    
    @reduce_dims
    def argmax(self, axis=None, split_every=None, out=None):
        if axis is None:
            result = int(super().argmax(axis=axis, split_every=split_every, out=out))
        else:
            result = self.like_data(super().argmax(axis=axis, split_every=split_every, out=out),
                                    title_suffix='_argmin_indices', reset_units=True, reset_quantity=True,
                                    check_dims=False)
        return result, locals().copy()
    
    def angle(self, deg=False):
        result = self.like_data(da.angle(self, deg=deg), reset_units=True,
                                reset_quantity=True, title_suffix='_angle', checkdims=True)
        if deg:
            result.units = 'degrees'
        else:
            result.units = 'radians'
        return result
    
    def conj(self):
        return self.like_data(super().conj(), reset_units=True,
                              reset_quantity=True, title_suffix='_conj', checkdims=True)

    def astype(self, dtype, **kwargs):
        return self.like_data(super().astype(dtype=dtype, **kwargs))

    def flatten(self):
        return self.like_data(super().flatten(), title_suffix='_raveled', check_dims=False)
    
    def ravel(self):
        return self.flatten()

    def clip(self, min=None, max=None):
        return self.like_data(super().clip(min=min, max=max), reset_quantity=True, title_suffix='_clipped')
    
    def compute_chunk_sizes(self):
        return self.like_data(super().compute_chunk_sizes())
    
    def cumprod(self, axis, dtype=None, out=None, method='sequential'):
        if axis is None:
            self = self.flatten()
            axis = 0
        
        return self.like_data(super().cumprod(axis=axis, dtype=dtype, out=out,
                                              method=method), title_suffix='_cumprod', reset_quantity=True)
        
    def cumsum(self, axis, dtype=None, out=None, method='sequential'):
        if axis is None:
            self = self.flatten()
            axis = 0
        
        return self.like_data(super().cumsum(axis=axis, dtype=dtype, out=out,
                                             method=method), title_suffix='_cumprod', reset_quantity=True)
    
    # What happens to the dimensions??
    def dot(self, other):
        return self.from_array(super().dot(other))

    def squeeze(self, axis=None):
        result = self.like_data(super().squeeze(axis=axis), title_prefix='Squeezed_', checkdims=False)
        if axis is None:
            shape_list = list(self.shape)
            axes = [i for i in range(self.ndim) if shape_list[i] == 1]
        elif isinstance(axis, int):
            axes = [axis]
        else:
            axes = list(axis)

        return self.__reduce_dimensions(result, axes, keepdims=False)
    
    def swapaxes(self, axis1, axis2):
        result = self.like_data(super().swapaxes(axis1, axis2), title_prefix='Swapped_axes_', checkdims=False)
        new_order = np.arange(self.ndim)
        new_order[axis1] = axis2
        new_order[axis2] = axis1
        # print(new_order)
        return self.__rearrange_axes(result, new_order)

    def transpose(self, *axes):
        result = self.like_data(super().transpose(*axes), title_prefix='Transposed_', checkdims=False)
        if not axes:
            new_axes_order = range(self.ndim)[::-1]
        elif len(axes) == 1 and isinstance(axes[0], Iterable):
            new_axes_order = axes[0]
        else:
            new_axes_order = axes
        return self.__rearrange_axes(result, new_axes_order)
    
    def round(self, decimals=0):
        return self.like_data(super().round(decimals=decimals), title_prefix='Rounded_')

    def reshape(self, *shape, merge_chunks=True):
        # This somehow adds an extra dimension at the end
        # Will come back to this
        warnings.warn('Dimensional information will be lost.\
                Please use fold unfold to combine dimensions')
        if len(shape) == 1 and isinstance(shape[0], Iterable):
            new_shape = shape[0]
        else: 
            new_shape = shape
        print(new_shape)
        return super().reshape(*new_shape, merge_chunks)
    
    @reduce_dims
    def prod(self, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
        if axis is None and not keepdims:
            result = float(super().prod())
        else:
            result = self.like_data(super().prod(axis=axis, dtype=dtype, keepdims=keepdims,
                                                 split_every=split_every, out=out), checkdims=False)
        return result, locals().copy()
    
    @reduce_dims
    def trace(self, offset=0, axis1=0, axis2=1, dtype=None):
        
        if self.ndim == 2:
            axes = None
            result = float(super().trace(offset=offset))
        
        else:
            axes = [axis1, axis2]
            result = self.like_data(super().trace(offset=offset, axis1=axis1,
                                                  axis2=axis2, dtype=None), title_prefix='Trace_', checkdims=False)
        local_args = locals().copy()
        local_args['axis'] = axes
        return result, local_args

    def repeat(self, repeats, axis=None):
        result = self.like_data(super().repeat(repeats=repeats, axis=axis),
                                title_prefix='Repeated_', checkdims=False)
        result._axes = {}
        for i, dim in self._axes.items():
            if axis != i:
                new_dim = dim.copy()
            else:
                new_dim = Dimension(np.repeat(dim.values, repeats=repeats),
                                    name=dim.name, quantity=dim.quantity,
                                    units=dim.units, dimension_type=dim.dimension_type)
            result.set_dimension(i, new_dim)
        
        return result
    
    @reduce_dims
    def moment(self, order, axis=None, dtype=None, keepdims=False, ddof=0, split_every=None, out=None):
        if axis is None and not keepdims:
            result = float(super().moment(order=order))
        else:
            result = self.like_data(super().moment(order=order,
                                                   axis=axis,
                                                   dtype=dtype, keepdims=keepdims,
                                                   ddof=0, split_every=split_every,
                                                   out=out), checkdims=False)
        return result, locals().copy()

    def persist(self, **kwargs):
        return self.like_data(super().persist(**kwargs), title_prefix='persisted_')
    
    def rechunk(self, chunks, threshold, block_size_limit, balance):
        return self.like_data(super().rechunk(chunks=chunks,
                                              threshold=threshold,
                                              block_size_limit=block_size_limit,
                                              balance=balance), title_prefix='Rechunked_')

    # Following methods are to be edited

    def adjust_axis(self, result, axis, title='', keepdims=False):
        if not keepdims:
            dim = 0
            dataset = self.from_array(result)
            if isinstance(axis, int):
                axis = [axis]

            # for ax, dimension in self._axes.items():
            #    if int(ax) not in axis:
            #        delattr(self, dimension.name)
            #        delattr(self, f'dim_{ax}')
            #        del self._axes[ax]

            for ax, dimension in self._axes.items():
                if int(ax) not in axis:
                    dataset.set_dimension(dim, dimension)
                    dim += 1
        else:
            dataset = self.like_data(result)
        dataset.title = title + self.title
        dataset.modality = f'sum axis {axis}'
        dataset.quantity = self.quantity
        dataset.source = self.source
        dataset.units = self.units

        return dataset

    def choose(self, choices):
        return self.like_data(super().choose(choices))

    def __abs__(self):
        # print(super().__abs__.__name__)
        return self.like_data(super().__abs__(), title_suffix='_absolute_value')

    def __add__(self, other):
        return self.like_data(super().__add__(other))

    def __radd__(self, other):
        return self.like_data(super().__radd__(other))

    def __and__(self, other):
        return self.like_data(super().__and__(other))

    def __rand__(self, other):
        return self.like_data(super().__rand__(other))

    def __div__(self, other):
        return self.like_data(super().__div__(other))

    def __rdiv__(self, other):
        return self.like_data(super().__rdiv__(other))

    def __gt__(self, other):
        return self.like_data(super().__gt__(other))

    def __ge__(self, other):
        return self.like_data(super().__ge__(other))

    def __invert__(self):
        return self.like_data(super().__invert__())

    def __lshift__(self, other):
        return self.like_data(super().__lshift__(other))

    def __rlshift__(self, other):
        return self.like_data(super().__rlshift__(other))

    def __lt__(self, other):
        return self.like_data(super().__lt__(other))

    def __le__(self, other):
        return self.like_data(super().__lt__(other))

    def __mod__(self, other):
        return self.like_data(super().__lshift__(other))

    def __rmod__(self, other):
        return self.like_data(super().__rmod__(other))

    def __mul__(self, other):
        return self.like_data(super().__mul__(other))

    def __rmul__(self, other):
        return self.like_data(super().__rmul__(other))

    def __ne__(self, other):
        return self.like_data(super().__ne__(other))

    def __neg__(self):
        return self.like_data(super().__neg__())

    def __or__(self, other):
        return self.like_data(super().__or__(other))

    def __ror__(self, other):
        return self.like_data(super().__ror__(other))

    def __pos__(self):
        return self.like_data(super().__pos__())

    def __pow__(self, other):
        return self.like_data(super().__pow__(other))

    def __rpow__(self, other):
        return self.like_data(super().__rpow__(other))

    def __rshift__(self, other):
        return self.like_data(super().__rshift__(other))

    def __rrshift__(self, other):
        return self.like_data(super().__rrshift__(other))

    def __sub__(self, other):
        return self.like_data(super().__sub__(other))

    def __rsub__(self, other):
        return self.like_data(super().__rsub__(other))

    def __truediv__(self, other):
        return self.like_data(super().__truediv__(other))

    def __rtruediv__(self, other):
        return self.like_data(super().__rtruediv__(other))

    def __floordiv__(self, other):
        return self.like_data(super().__floordiv__(other))

    def __rfloordiv__(self, other):
        return self.like_data(super().__rfloordiv__(other))

    def __xor__(self, other):
        return self.like_data(super().__xor__(other))

    def __rxor__(self, other):
        return self.like_data(super().__rxor__(other))

    def __matmul__(self, other):
        return self.like_data(super().__matmul__(other))

    def __rmatmul__(self, other):
        return self.like_data(super().__rmatmul__(other))

    def __array_ufunc__(self, numpy_ufunc, method, *inputs, **kwargs):
        out = kwargs.get("out", ())

        if method == "__call__":
            if numpy_ufunc is np.matmul:
                from dask.array.routines import matmul

                # special case until apply_gufunc handles optional dimensions
                return self.like_data(matmul(*inputs, **kwargs))
            if numpy_ufunc.signature is not None:
                from dask.array.gufunc import apply_gufunc

                return self.like_data(apply_gufunc(
                    numpy_ufunc, numpy_ufunc.signature, *inputs, **kwargs))
            if numpy_ufunc.nout > 1:
                from dask.array import ufunc

                try:
                    da_ufunc = getattr(ufunc, numpy_ufunc.__name__)
                except AttributeError:
                    return NotImplemented
                return self.like_data(da_ufunc(*inputs, **kwargs))
            else:
                return self.like_data(dask.array.core.elemwise(numpy_ufunc, *inputs, **kwargs))
        elif method == "outer":
            from dask.array import ufunc

            try:
                da_ufunc = getattr(ufunc, numpy_ufunc.__name__)
            except AttributeError:
                return NotImplemented
            return self.like_data(da_ufunc.outer(*inputs, **kwargs))
        else:
            return NotImplemented


def convert_hyperspy(s):
    """
    imports a hyperspy signal object into sidpy.Dataset

    Parameters
    ----------
    s: hyperspy dataset

    Return
    ------
    dataset: sidpy.Dataset
    """
    try:
        import hyperspy.api as hs
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Hyperspy is not installed")

    if not isinstance(s, (hs.signals.Signal1D, hs.signals.Signal2D)):
        raise TypeError('This is not a hyperspy signal object')
    dataset = Dataset.from_array(s, name=s.metadata.General.title)
    # Add dimension info
    axes = s.axes_manager.as_dictionary()

    if isinstance(s, hs.signals.Signal1D):
        if s.data.ndim < 2:
            dataset.data_type = 'spectrum'
        elif s.data.ndim > 1:
            if s.data.ndim == 2:
                dataset = Dataset.from_array(np.expand_dims(s, 2), title=s.metadata.General.title)
                dataset.set_dimension(2, Dimension([0], name='y', units='pixel',
                                                   quantity='distance', dimension_type='spatial'))
            dataset.data_type = DataType.SPECTRAL_IMAGE
        for key, axis in axes.items():
            if axis['navigate']:
                dimension_type = 'spatial'
            else:
                dimension_type = 'spectral'
            dim_array = np.arange(axis['size']) * axis['scale'] + axis['offset']
            if axis['units'] == '':
                axis['units'] = 'frame'
            dataset.set_dimension(int(key[-1]), Dimension(dim_array, name=axis['name'], units=axis['units'],
                                                          quantity=axis['name'], dimension_type=dimension_type))

    elif isinstance(s, hs.signals.Signal2D):
        if s.data.ndim < 4:
            if s.data.ndim == 2:
                dataset.data_type = 'image'
            elif s.data.ndim == 3:
                dataset.data_type = 'image_stack'
            for key, axis in axes.items():
                if axis['navigate']:
                    dimension_type = 'temporal'
                else:
                    dimension_type = 'spatial'
                dim_array = np.arange(axis['size']) * axis['scale'] + axis['offset']
                if axis['units'] == '':
                    axis['units'] = 'pixel'
                dataset.set_dimension(int(key[-1]), Dimension(dim_array, name=axis['name'], units=axis['units'],
                                                              quantity=axis['name'],
                                                              dimension_type=dimension_type))
        elif s.data.ndim == 4:
            dataset.data_type = 'IMAGE_4D'
            for key, axis in axes.items():
                if axis['navigate']:
                    dimension_type = 'spatial'
                else:
                    dimension_type = 'reciprocal'
                dim_array = np.arange(axis['size']) * axis['scale'] + axis['offset']
                dataset.set_dimension(int(key[-1]), Dimension(dim_array, name=axis['name'], units=axis['units'],
                                                              quantity=axis['name'],
                                                              dimension_type=dimension_type))
    dataset.metadata = dict(s.metadata)
    dataset.original_metadata = dict(s.original_metadata)
    dataset.title = dataset.metadata['General']['title']
    dataset.units = dataset.metadata['Signal']['quantity '].split('(')[-1][:-1]
    dataset.quantity = dataset.metadata['Signal']['quantity '].split('(')[0]
    dataset.source = 'hyperspy'
    return dataset
