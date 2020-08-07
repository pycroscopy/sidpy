# -*- coding: utf-8 -*-
"""
Utilities that assist in writing USID related data to HDF5 files

Created on Thu Jul  7 21:14:25 2020

@author: Gerd Duscher, Suhas Somnath
"""


from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import numpy as np
from ..base.string_utils import validate_single_string_arg

__all__ = ['Dimension']

if sys.version_info.major == 3:
    unicode = str


class Dimension(object):
    """
    ..autoclass::Dimension
    """

    def __init__(self, name, values, quantity='generic', units='generic',  dimension_type='generic'):
        """
        Simple object that describes a dimension in a dataset by its name, units, and values
        Parameters
        ----------
        name : str or unicode
            Name of the dimension. For example 'X'
        quantity : str or unicode
            Quantity for this dimension. For example: 'Length'
        units : str or unicode
            Units for this dimension. For example: 'um'
        values : array-like or int
            Values over which this dimension was varied. A linearly increasing set of values will be generated if an
            integer is provided instead of an array.
        dimension_type : str or unicode for example: 'spectral' or 'spatial', 'time', 'frame', 'reciprocal'
            This will determine how the data are visualized. 'spatial' are image dimensions.
            'spectral' indicate spectroscopy data dimensions.
        """

        self.name = name
        self.values= values

        self.quantity = quantity
        self.units = units
        self.dimension_type =dimension_type


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
        self._dimension_type  = validate_single_string_arg(value, 'dimension_type')

    @property
    def values(self):
        return self._values
    @values.setter
    def values(self, values):
        if isinstance(values, int):
            if values < 1:
                raise ValueError('values should at least be specified as a positive integer')
            values = np.arange(values)
        if not isinstance(values, (np.ndarray, list, tuple)):
            raise TypeError('values should be array-like')
        values = np.array(values)
        if values.ndim > 1:
            raise ValueError('Values for dimension: {} are not 1-dimensional'.format(self.name))

        self._values = values

    def copy(self):
        return Dimension(self.name, self.values, self.quantity, self.units, self.dimension_type)

    def __repr__(self):
        return '{} - {} ({}): {}'.format(self.name, self.quantity, self.units, self.values)

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