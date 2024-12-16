# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import unittest
import warnings

import numpy as np
from numpy.testing import assert_array_equal
from sidpy.sid.dimension import Dimension

sys.path.insert(0, "../../sidpy/")

if sys.version_info.major == 3:
    unicode = str


class TestDimension(unittest.TestCase):

    def test_values_as_array(self):
        name = 'Bias'
        values = np.random.rand(5)

        descriptor = Dimension(values, name)
        for expected, actual in zip([name, values],
                                    [descriptor.name, descriptor.values]):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))

    def test_values_as_length(self):
        name = 'Bias'
        units = 'V'
        values = np.arange(5)

        descriptor = Dimension(len(values), name, units=units)
        for expected, actual in zip([name, units],
                                    [descriptor.name, descriptor.units]):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))
        self.assertTrue(np.allclose(values, descriptor.values))

    def test_copy(self):
        name = 'Bias'
        units = 'V'
        values = np.arange(5)

        descriptor = Dimension(values, name, units=units)
        copy_descriptor = descriptor.copy()

        for expected, actual in zip([copy_descriptor.name, copy_descriptor.units],
                                    [descriptor.name, descriptor.units]):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))
        self.assertTrue(np.allclose(copy_descriptor.values, descriptor.values))
        copy_descriptor.units = 'eV'
        self.assertFalse(copy_descriptor.units == descriptor.units)


    def test_repr(self):
        name = 'Bias'
        values = np.arange(5)

        descriptor = Dimension(values, name)
        actual = '{}'.format(descriptor)
        quantity = 'generic'
        units = 'generic'
        expected = '{}:  {} ({}) of size {}'.format(name, quantity, units, values.shape)
        self.assertEqual(actual, expected)

    def test_change_name(self):
        name = 'Bias'
        values = np.arange(5)

        descriptor = Dimension(values, name)
        with self.assertRaises(AttributeError):
            descriptor.name = 'Voltage'

    def test_inequality_req_inputs(self):
        name = 'X'
        quantity = "Length"
        units = 'nm'

        self.assertTrue(Dimension(5, name) == Dimension(5, name))
        self.assertFalse(Dimension(5, 'Y') == Dimension(5, name))
        self.assertFalse(Dimension(4, name) == Dimension(5, name))

        self.assertTrue(
            Dimension(5, units=units) == Dimension(5, units=units))
        self.assertFalse(
            Dimension(5, units='pm') == Dimension(5, units=units))

        self.assertTrue(
            Dimension(5, quantity=quantity) == Dimension(5, quantity=quantity))
        self.assertFalse(
            Dimension(5, quantity='Bias') == Dimension(5, quantity=quantity))

        self.assertFalse(
            Dimension(np.arange(5)) == Dimension(np.arange(5) + 1))

    def test_dimensionality(self):
        vals = np.ones((2, 2))
        expected = 'Dimension can only be 1 dimensional'
        with self.assertRaises(Exception) as context:
            _ = Dimension(vals, "x", )
        self.assertTrue(expected in str(context.exception))

    def test_info(self):
        expected = "X - Bias (mV): [0. 1. 2. 3. 4.]"
        dim = Dimension(np.arange(5), "X", "Bias", "mV")
        self.assertTrue(dim.info, expected)

    def test_values_smaller_than_min_size(self):
        with self.assertRaises(TypeError) as context:
            _ = Dimension(0, name="x")
        self.assertTrue("When specifying the size of a Dimension, values "
                        "should at be integers > 1" in str(context.exception))

    def test_empty_array_values(self):
        with self.assertRaises(TypeError) as context:
            _ = Dimension([], name="x")
        self.assertTrue("When specifying values over which a parameter is "
                        "varied, values should not be an empty array"
                        "" in str(context.exception))

    def test_dimension_size_1(self):
        dim = Dimension(1)
        self.assertIsInstance(dim, Dimension)
        assert_array_equal(np.array(dim), [0])

    def test_single_valued_dimension(self):
        dim = Dimension([1.23])
        self.assertIsInstance(dim, Dimension)
        assert_array_equal(np.array(dim), [1.23])

    def test_conv2arr_values(self):
        arr = np.arange(5)
        vals = [5, arr, arr.tolist(), tuple(arr)]
        vals_expected = arr
        for v in vals:
            dim = Dimension(v, "x")
            self.assertIsInstance(dim, Dimension)
            assert_array_equal(np.array(dim), vals_expected)

    def test_dimension_type(self):
        dim_types = ["spatial", "Spatial", "reciprocal", "Reciprocal",
                     "spectral", "Spectral", "temporal", "Temporal",
                     "frame", "Frame", "time", "Time", "stack", "Stack"]
        dim_vals_expected = [1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4]
        dim_names_expected = ["SPATIAL", "SPATIAL", "RECIPROCAL", "RECIPROCAL",
                              "SPECTRAL", "SPECTRAL", "TEMPORAL", "TEMPORAL",
                              "TEMPORAL", "TEMPORAL", "TEMPORAL", "TEMPORAL",
                              "TEMPORAL", "TEMPORAL"]
        for dt, dv, dn in zip(dim_types, dim_vals_expected, dim_names_expected):
            dim = Dimension(5, "x", dimension_type=dt)
            self.assertEqual(dim.dimension_type.value, dv)
            self.assertEqual(dim.dimension_type.name, dn)

    def test_dimension_type(self):
        dim_types = ["spatial", "Spatial", "reciprocal", "Reciprocal",
                     "spectral", "Spectral", "temporal", "Temporal",
                     "frame", "Frame", "time", "Time", "stack", "Stack"]
        dim_vals_expected = [1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4]
        dim_names_expected = ["SPATIAL", "SPATIAL", "RECIPROCAL", "RECIPROCAL",
                              "SPECTRAL", "SPECTRAL", "TEMPORAL", "TEMPORAL",
                              "TEMPORAL", "TEMPORAL", "TEMPORAL", "TEMPORAL",
                              "TEMPORAL", "TEMPORAL"]
        for dt, dv, dn in zip(dim_types, dim_vals_expected, dim_names_expected):
            dim = Dimension(5, "x", dimension_type=dt)
            self.assertEqual(dim.dimension_type.value, dv)
            self.assertEqual(dim.dimension_type.name, dn)

    def test_unknown_dimension_type(self):
        dim_type = "bad_name"
        expected_wrn = ["Supported dimension types for plotting are only: [",
                        "Setting DimensionType to UNKNOWN"]
        with warnings.catch_warnings(record=True) as w:
            _ = Dimension(5, "x", dimension_type=dim_type)
        self.assertTrue(expected_wrn[0] in str(w[0].message))
        self.assertTrue(expected_wrn[1] in str(w[1].message))

    def test_add(self):
        name = 'Bias'
        units = 'V'
        values = np.arange(5)

        descriptor = Dimension(values, name, units=units)
        descriptor = descriptor + 3.
        self.assertIsInstance(descriptor, Dimension)


if __name__ == '__main__':
    unittest.main()
