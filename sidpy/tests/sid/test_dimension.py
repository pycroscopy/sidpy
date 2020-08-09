# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys
import numpy as np

sys.path.append("../../sidpy/")
from sidpy.sid.dimension import Dimension

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