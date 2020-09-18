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

__all__ = ['Dimension']

if sys.version_info.major == 3:
    unicode = str

# todo: Consider extending numpy.ndarray instead of generic python object


class DimensionTypes(Enum):
    UNKNOWN = -1
    SPATIAL = 1
    RECIPROCAL = 2
    SPECTRAL = 3
    TEMPORAL = 4


class Dimension(np.ndarray):
    """
    """

    def __new__(cls, values, name='none', quantity='generic', units='generic', dimension_type=DimensionTypes.UNKNOWN):
        """
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

        :param values: array-like or int
            Values over which this dimension was varied. A linearly increasing
            set of values will be generated if an integer is provided instead
            of an array.
        :param name: str or unicode
            Name of the dimension. For example 'X'
        :param quantity:   str or unicode
            Quantity for this dimension. For example: 'Length'
        :param units: str or unicode
            Units for this dimension. For example: 'um'
        :param dimension_type: Dimension_Type: i.e: 'spectral', 'spatial', 'reciprocal', 'temporal' or 'UNKNOWN'
            This will determine how the data are visualized. 'spatial' are
            image dimensions. 'spectral' indicate spectroscopy data dimensions.
        """

        new_dim = np.asarray(values).view(cls)
        new_dim.name = name
        new_dim.quantity = quantity
        new_dim.units = units
        new_dim.dimension_type = dimension_type
        print(new_dim.name)
        return new_dim

    def __repr__(self):
        return '{} - {} ({}): {}'.format(self.name, self.quantity, self.units,
                                         self.values)

    @property
    def info(self):
        return '{} - {} ({}): {}'.format(self.name, self.quantity, self.units,
                                         self.values)

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
        return self

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
