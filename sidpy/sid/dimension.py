# -*- coding: utf-8 -*-
"""
Utilities that assist in writing USID related data to HDF5 files

Created on Thu Jul  7 21:14:25 2020

@author: Gerd Duscher, Suhas Somnath
"""

from __future__ import division, print_function, unicode_literals, \
    absolute_import
import sys
import numpy as np
from enum import Enum
from sidpy.base.string_utils import validate_single_string_arg

__all__ = ['Dimension', 'DimensionTypes']

if sys.version_info.major == 3:
    unicode = str

# todo: Consider extending numpy.ndarray instead of generic python object


class DimensionTypes(Enum):
    UNKNOWN = -1
    SPATIAL = 1
    RECIPROCAL = 2
    SPECTRAL = 3
    TEMPORAL = 4


class Dimension(object):
    """
    """

    def __init__(self, name, values, quantity='generic', units='generic',
                 dimension_type=DimensionTypes.UNKNOWN):
        """
        Simple object that describes a dimension in a dataset by its name,
        units, and values

        Parameters
        ----------
        name : str or unicode
            Name of the dimension. For example 'X'
        quantity : str or unicode
            Quantity for this dimension. For example: 'Length'
        units : str or unicode
            Units for this dimension. For example: 'um'
        values : array-like or int
            Values over which this dimension was varied. A linearly increasing
            set of values will be generated if an integer is provided instead
            of an array.
        dimension_type : str or unicode for example: 'spectral' or 'spatial',
            'time', 'frame', 'reciprocal'
            This will determine how the data are visualized. 'spatial' are
            image dimensions. 'spectral' indicate spectroscopy data dimensions.

        Attributes
        ----------
        self.name : str
            Name of the dimension
        self.quantity : str
            Physical quantity. E.g. - current
        self.units : str
            Physical units. E.g. - amperes
        self.dimension_type : enum
            Type of dimension. E.g. - Spectral, Spatial, etc.
        self.values : array-like
            Values over which this dimension was varied
        """

        self.name = name
        self.values = values

        self.quantity = quantity
        self.units = units
        self.dimension_type = dimension_type

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = validate_single_string_arg(value, 'name')

    @property
    def quantity(self):
        return self._quantity

    @quantity.setter
    def quantity(self, value):
        self._quantity = validate_single_string_arg(value, 'quantity')

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        self._units = validate_single_string_arg(value, 'units')

    @property
    def dimension_type(self):
        return self._dimension_type

    @dimension_type.setter
    def dimension_type(self, value):
        if isinstance(value, DimensionTypes):
            self._dimension_type = value
        else:
            dimension_type = validate_single_string_arg(value, 'dimension_type')

            if dimension_type.upper() in DimensionTypes._member_names_:
                self._dimension_type = DimensionTypes[dimension_type.upper()]
            elif dimension_type.lower() in ['frame', 'time', 'stack']:
                self._dimension_type = DimensionTypes.TEMPORAL
            else:
                self._dimension_type = DimensionTypes.UNKNOWN
                print('Supported dimension_types for plotting are only: ', DimensionTypes._member_names_)
                print('Setting DimensionTypes to UNKNOWN')

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        if isinstance(values, int):
            if values < 1:
                raise ValueError('values should at least be specified as a '
                                 'positive integer')
            values = np.arange(values)
        if not isinstance(values, (np.ndarray, list, tuple)):
            raise TypeError('values should be array-like')
        values = np.array(values)
        if values.ndim > 1:
            raise ValueError('Values for dimension: {} are not 1-dimensional'
                             ''.format(self.name))

        self._values = values

    def copy(self):
        """
        Returns a copy of this Dimension

        Returns
        -------
        sidpy.Dimension
            Copy of this sidpy.Dimension
        """
        return Dimension(self.name, self.values, self.quantity, self.units,
                         self.dimension_type)

    def __repr__(self):
        return '{} - {} ({}): {}'.format(self.name, self.quantity, self.units,
                                         self.values)

    def __eq__(self, other):
        if isinstance(other, Dimension):
            if self.name != other.name:
                return False
            if self.units != other.units:
                return False
            if self.quantity != other.quantity:
                return False
            if len(self.values) != len(other.values):
                return False
            if not np.allclose(self.values, other.values):
                return False
        return True
