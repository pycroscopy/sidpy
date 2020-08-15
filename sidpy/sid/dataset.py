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
import string
import dask.array as da
from .dimension import Dimension
from ..base.num_utils import get_slope


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
                    (the axes are dimension datsets with name, label, units, and 'dimension_type' attributes).

    -attrs: dictionary of additional metadata
    -orginal_metadata: dictionary of original metadata of file,

    -labels: returns labels of all dimensions.

    functions:
    set_dimension(axis, dimensions): set a Dimension to a specific axis
    """

    def __init__(self, *args, **kwargs):
        super(Dataset, self).__init__(*args, **kwargs)

    @classmethod
    def from_array(cls, x, chunks=None, name=None, lock=False):
        """

        Parameters
        ----------
        x
        chunks
        name
        lock

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
        cls.units = ''
        cls.title = ''
        cls.quantity = 'generic'

        cls.modality = ''
        cls.source = ''
        cls.data_descriptor = ''

        cls.axes = {}
        for dim in range(cls.ndim):
            # TODO: add parent to dimension to set attribute if name changes
            cls.labels.append(string.ascii_lowercase[dim])
            cls.set_dimension(dim,
                              Dimension(string.ascii_lowercase[dim],
                                        np.arange(cls.shape[dim])))
        cls.attrs = {}
        cls.group_attrs = {}
        cls.original_metadata = {}
        return cls

    def like_data(self, data,  name=None, lock=False):
        """

        Parameters
        ----------
        data
        name
        lock

        Returns
        -------

        """

        new_data = self.from_array(data, chunks=None, name=None, lock=False)

        new_data.data_type = self.data_type
        new_data.units = self.units
        new_data.title = self.title + "_new"
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
                # assumes the axis scale is equidistant
                # TODO: test below using get_slope()
                scale = self.axes[dim].values[1] - self.axes[dim].values[1]
                axis = self.axes[dim].copy()
                axis.values = np.arange(new_data.shape[dim])*scale
                new_data.set_dimension(dim, axis)

        new_data.attrs = dict(self.attrs).copy()
        new_data.group_attrs = {}  # dict(self.group_attrs).copy()
        new_data.original_metadata = {}
        return new_data


    """@classmethod
    def from_hdf5(cls, dset, chunks=None, name=None, lock=False):

        # determine chunks, guessing something reasonable if user does not
        # specify
        chunks = get_chunks(np.array(dset), chunks)

        # create vanilla dask array
        darr = da.from_array(np.array(dset), chunks=chunks, name=name, lock=lock)

        # view as sub-class
        cls = view_subclass(darr, cls)

        if 'title' in dset.attrs:
            cls.title = dset.attrs['title']
        else:
            cls.title = dset.name

        if 'units' in dset.attrs:
            cls.units = dset.attrs['units']
        else:
            cls.units = 'generic'

        if 'quantity' in dset.attrs:
            cls.quantity = dset.attrs['quantity']
        else:
            cls.quantity = 'generic'

        if 'data_type' in dset.attrs:
            cls.data_type = dset.attrs['data_type']
        else:
            cls.data_type = 'generic'

        #TODO: mdoality and source not yet properties
        if 'modality' in dset.attrs:
            cls.modality = dset.attrs['modality']
        else:
            cls.modality = 'generic'

        if 'source' in dset.attrs:
            cls.source = dset.attrs['source']
        else:
            cls.source = 'generic'

        cls.axes ={}

        for dim in range(np.array(dset).ndim):
            #print(dim, dset.dims[dim].label)
            #print(dset.dims[dim][0][0])
            dim_dict = dict(dset.parent[dset.dims[dim].label].attrs)
            #print(dset.dims[dim].label, np.array(dset.dims[dim][0]))
            #print(dset.parent[dset.dims[0].label][()])
            #print(dim_dict['quantity'], dim_dict['units'], dim_dict['dimension_type'])
            cls.set_dimension(dim, Dimension(dset.dims[dim].label, np.array(dset.parent[dset.dims[dim].label][()]),
                                                    dim_dict['quantity'], dim_dict['units'],
                                                    dim_dict['dimension_type']))
        cls.attrs = dict(dset.attrs)

        cls.original_metadata = {}
        if 'original_metadata' in dset.parent:
            cls.original_metadata = dict(dset.parent['original_metadata'].attrs)


        return cls

    def to_hdf5(self, h5_group):
        if  self.title.strip() == '':
            main_data_name = 'nDim_Data'
        else:
            main_data_name = self.title
        print(h5_group)
        print(h5_group.keys())

        print(main_data_name)

        dset = write_main_dataset(h5_group, np.array(self), main_data_name,
                                 self.quantity, self.units, self.data_type, self.modality,
                                 self.source, self.axes, verbose=False)
        print('d',dset)

        for key, item in self.attrs.items():
            #TODO: Check item to be simple
            dset.attrs[key] = item

        original_group = h5_group.create_group('original_metadata')
        for key, item in self.original_metadata.items():
            original_group.attrs[key] = item

        if hasattr(self, 'aberrations'):
            aberrations_group = h5_group.create_group('aberrations')
            for key, item in self.aberrations.items():
                aberrations_group.attrs[key] = item

        if hasattr(self, 'annotations'):
            annotations_group = h5_group.create_group('annotations')
            for key, item in self.annotations.items():
                annotations_group.attrs[key] = item
    """
    def copy(self):
        """

        Returns
        -------

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

    def set_dimension(self,dim, dimension):
        """

        Parameters
        ----------
        dim
        dimension

        Returns
        -------

        """
        # TODO: Check whether dimension valid
        setattr(self, dimension.name,dimension)
        setattr(self, 'dim_{}'.format(dim), dimension)

        self.axes[dim] = dimension

    def get_extent(self, dimensions):
        """
        get image extend as neeed i.e. in matplotlib's imshow function.
        This function works for equi or non-equi spaced axes

        Parameters
        ----------
        dimensions

        Returns
        -------

        """
        extend = []
        for i, dim in enumerate(dimensions):
            temp = self.axes[dim].values
            start = temp[0] - (temp[1] - temp[0])/2
            end = temp[-1] - (temp[-1] - temp[-2])/2
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
        if isinstance(value, str) :
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
