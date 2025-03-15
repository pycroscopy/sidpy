# -*- coding: utf-8 -*-
"""
Objects that represent dimensions or axes of scientific datasets

Created on Thu Jul  7 21:14:25 2020

@author: Gerd Duscher, Suhas Somnath
"""

from __future__ import division, print_function, unicode_literals, \
    absolute_import
from warnings import warn
import sys
import numpy as np
from enum import Enum
from sidpy.base.string_utils import validate_single_string_arg
import copy

__all__ = ['Dimension', 'DimensionType']

if sys.version_info.major == 3:
    unicode = str


class DimensionType(Enum):
    """
    Physical type of Dimension object. This information will be used for
    visualization and processing purposes.
    """
    UNKNOWN = -1
    SPATIAL = 1
    RECIPROCAL = 2
    SPECTRAL = 3
    TEMPORAL = 4
    CHANNEL = 5
    POINT_CLOUD = 6


class Dimension(np.ndarray):
    """
    """

    def __new__(cls, values, name='none', quantity='generic', units='generic',
                dimension_type=DimensionType.UNKNOWN, *args, **kwargs):
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
        dimension_type : str or sidpy.sid.dimension.DimensionType
            For example: 'spectral', 'spatial', 'reciprocal', 'channel', or 'UNKNOWN',
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
        if isinstance(values, int):
            if values < 1:
                raise TypeError("When specifying the size of a Dimension, "
                                "values should at be integers > 1")
            values = np.arange(values)
        elif len(np.array(values)) < 1:
            raise TypeError("When specifying values over which a parameter is "
                            "varied, values should not be an empty array")
        if np.array(values).ndim != 1:
            raise ValueError('Dimension can only be 1 dimensional')
        new_dim = np.asarray(values, dtype=float).view(cls)
        new_dim._name = validate_single_string_arg(name, 'name')
        new_dim.quantity = quantity
        new_dim.units = units
        new_dim.dimension_type = dimension_type
        return new_dim

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self._name = validate_single_string_arg(getattr(obj, '_name', 'generic'), 'name')
        self.quantity = getattr(obj, '_quantity', 'generic')
        self.units = getattr(obj, '_units', 'generic')
        self.dimension_type = getattr(obj, '_dimension_type', 'UNKNOWN')

    def __array_wrap__(self, out_arr, context=None):
        # just call the parent
        values = super(Dimension, self).__array_wrap__(self, out_arr, context)
        new_dim = Dimension(values, name=self.name, quantity=self.quantity, units=self.units)
        # return correct values
        return new_dim
    
   
    def __repr__(self):
        return '{}:  {} ({}) of size {}'.format(self.name, self.quantity, self.units, self.shape)

    def __str__(self):
        return '{}:  {} ({}) of size {}'.format(self.name, self.quantity, self.units, self.shape)

    # def __copy__(self):
    #     new_dim = Dimension(np.array(self), name=self.name, quantity=self.quantity, units=self.units)
    #     new_dim.dimension_type = self.dimension_type
    #     return new_dim

    def __copy__(self):
        # Create a new instance of Dimension
        new_instance = Dimension(
            copy.copy(self.values),
            copy.copy(self.name),
            copy.copy(self.quantity),
            copy.copy(self.units),
            copy.copy(self.dimension_type)
        )

        return new_instance

    def __deepcopy__(self, memo):
        # For now this is what chatGPT came up with and it does not break any tests

        # Create a new instance of Dimension
        new_instance = Dimension(
            copy.deepcopy(self.values, memo),
            copy.deepcopy(self.name, memo),
            copy.deepcopy(self.quantity, memo),
            copy.deepcopy(self.units, memo),
            copy.deepcopy(self.dimension_type, memo)
        )

        return new_instance

    # TODO: Implement equality

    # TODO: Find out how to get rid of this
    def copy(self):
        # Not sure why __copy__() would not be called by itself
        new_dim = self.__copy__()
        return new_dim

    @property
    def info(self):
        return '{} - {} ({}): {}'.format(self.name, self.quantity, self.units,
                                         self.values)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        raise AttributeError("Cannot change the name of the dimension. "
                             "If the dimension is associated with the dataset, please try "
                             "dataset.rename_dimension")

    #     # self._name = validate_single_string_arg(value, 'name')

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
        if isinstance(value, DimensionType):
            self._dimension_type = value
        else:
            dimension_type = validate_single_string_arg(value, 'dimension_type')

            if dimension_type.upper() in [member.name for member in DimensionType]:
                self._dimension_type = DimensionType[dimension_type.upper()]
            elif dimension_type.lower() in ['frame', 'time', 'stack']:
                self._dimension_type = DimensionType.TEMPORAL
            else:
                self._dimension_type = DimensionType.UNKNOWN
                warn('Supported dimension types for plotting are only: {}'
                     ''.format([member.name for member in DimensionType]))
                warn('Setting DimensionType to UNKNOWN')

    @property
    def values(self):
        return np.array(self)

    # @values.setter
    # def values(self, value):
    #     isinstance(np.ndarray)

    @property
    def slope(self):
        if len(self.values) < 2:
            return 0
        derivative = np.diff(self.values)
        slope = np.average(derivative)
        if derivative.std() > slope/1000:
            warn(f'Slope of dimension {self.name} is not constant')
        return slope

    def __eq__(self, other):
        if not isinstance(other, Dimension):
            return False
        if self.name != other.name:
            return False
        if self.units != other.units:
            return False
        if self.quantity != other.quantity:
            return False
        if len(self.values) != len(other):
            return False
        if not (np.array(self) == np.array(other)).all():
            return False
        if not (self.values == other.values).all():
            return False
        return True
