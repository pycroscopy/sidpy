# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from sidpy.sid.dimension import Dimension

sys.path.append("../../sidpy/")

if sys.version_info.major == 3:
    unicode = str


class TestDimension(unittest.TestCase):

    def test_values_as_array(self):
        name = 'Bias'
        values = np.random.rand(5)

        descriptor = Dimension(name, values)
        for expected, actual in zip([name, values],
                                    [descriptor.name, descriptor.values]):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))

    def test_values_as_length(self):
        name = 'Bias'
        units = 'V'
        values = np.arange(5)

        descriptor = Dimension(name, len(values), units=units)
        for expected, actual in zip([name, units],
                                    [descriptor.name, descriptor.units]):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))
        self.assertTrue(np.allclose(values, descriptor.values))

    def test_copy(self):
        name = 'Bias'
        units = 'V'
        values = np.arange(5)

        descriptor = Dimension(name, len(values), units=units)
        copy_descriptor = descriptor.copy()
        for expected, actual in zip([copy_descriptor.name, copy_descriptor.units],
                                    [descriptor.name, descriptor.units]):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))
        self.assertTrue(np.allclose(copy_descriptor.values, descriptor.values))
        copy_descriptor.units ='eV'
        copy_descriptor.name = 'energy'
        for expected, actual in zip([copy_descriptor.name, copy_descriptor.units],
                                    [descriptor.name, descriptor.units]):
            self.assertTrue(np.all([x != y for x, y in zip(expected, actual)]))
        copy_descriptor.values = descriptor.values+1
        self.assertFalse(np.allclose(copy_descriptor.values, descriptor.values))

    def test_repr(self):
        name = 'Bias'
        values = np.arange(5)

        descriptor = Dimension(name, len(values))
        actual = '{}'.format(descriptor)

        quantity = 'generic'
        units = 'generic'
        expected = '{} - {} ({}): {}'.format(name, quantity, units, values)
        self.assertEqual(actual, expected)

    def test_equality(self):
        name = 'Bias'

        dim_1 = Dimension(name, [0, 1, 2, 3, 4])
        dim_2 = Dimension(name, np.arange(5, dtype=np.float32))
        self.assertEqual(dim_1, dim_2)

    def test_inequality_req_inputs(self):
        name = 'Bias'

        self.assertNotEqual(Dimension(name, [0, 1, 2, 3]),
                            Dimension(name, [0, 1, 2, 4]))

        self.assertNotEqual(Dimension('fdfd', [0, 1, 2, 3]),
                            Dimension(name, [0, 1, 2, 3]))

        self.assertNotEqual(Dimension(name, [0, 1, 2]),
                            Dimension(name, [0, 1, 2, 3]))

    def test_dimensionality(self):
        vals = np.ones((2, 2))
        expected = "Values for dimension: x are not 1-dimensional"
        with self.assertRaises(Exception) as context:
            _ = Dimension("x", vals)
        self.assertTrue(expected in str(context.exception))

    def test_nonposint_values(self):
        vals = [-1, .5]
        expected = ["values should at least be specified as a positive integer",
                    "values should be array-like"]
        for v, e in zip(vals, expected):
            with self.assertRaises(Exception) as context:
                _ = Dimension("x", v)
            self.assertTrue(e in str(context.exception))

    def test_conv2arr_values(self):
        arr = np.arange(5)
        vals = [5, arr, arr.tolist(), tuple(arr)]
        vals_expected = arr
        for v in vals:
            dim = Dimension("x", v)
            self.assertIsInstance(dim.values, np.ndarray)
            assert_array_equal(dim.values, vals_expected)

    def test_dimension_type(self):
        dim_types = ["spatial", "Spatial", "reciprocal", "Reciprocal",
                     "spectral", "Spectral", "temporal", "Temporal",
                     "frame", "Frame",  "time", "Time", "stack", "Stack"]
        dim_vals_expected = [1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4]
        dim_names_expected = ["SPATIAL", "SPATIAL", "RECIPROCAL", "RECIPROCAL",
                              "SPECTRAL", "SPECTRAL", "TEMPORAL", "TEMPORAL",
                              "TEMPORAL", "TEMPORAL", "TEMPORAL", "TEMPORAL",
                              "TEMPORAL", "TEMPORAL"]
        for dt, dv, dn in zip(dim_types, dim_vals_expected, dim_names_expected):
            dim = Dimension("x", 5, dimension_type=dt)
            self.assertEqual(dim.dimension_type.value, dv)
            self.assertEqual(dim.dimension_type.name, dn)