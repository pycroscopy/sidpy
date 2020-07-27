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
from sidpy.io import write_utils

if sys.version_info.major == 3:
    unicode = str


class TestCleanStringAtt(unittest.TestCase):
            
    def test_float(self):
        expected = 5.321
        self.assertEqual(expected, write_utils.clean_string_att(expected))

    def test_str(self):
        expected = 'test'
        self.assertEqual(expected, write_utils.clean_string_att(expected))

    def test_num_array(self):
        expected = [1, 2, 3.456]
        self.assertEqual(expected, write_utils.clean_string_att(expected))

    def test_str_list(self):
        expected = ['a', 'bc', 'def']
        returned = write_utils.clean_string_att(expected)
        expected = np.array(expected, dtype='S')
        for exp, act in zip(expected, returned):
            self.assertEqual(exp, act)

    def test_str_tuple(self):
        expected = ('a', 'bc', 'def')
        returned = write_utils.clean_string_att(expected)
        expected = np.array(expected, dtype='S')
        for exp, act in zip(expected, returned):
            self.assertEqual(exp, act)


class TestGetSlope(unittest.TestCase):

    def test_linear(self):
        expected = 0.25
        actual = write_utils.get_slope(np.arange(-1, 1, expected))
        self.assertEqual(expected, actual)

    def test_linear_dirty(self):
        # When reading from HDF5, rounding errors can result in minor variations in the diff
        expected = 0.25E-9
        vector = np.arange(-1E-9, 1E-9, expected)
        round_error = np.random.rand(vector.size) * 1E-14
        vector += round_error
        actual = write_utils.get_slope(vector, tol=1E-3)
        self.assertAlmostEqual(expected, actual)

    def test_invalid_tolerance(self):
        with self.assertRaises(TypeError):
            _ = write_utils.get_slope(np.sin(np.arange(4)), tol="hello")

    def test_non_linear(self):
        with self.assertRaises(ValueError):
            _ = write_utils.get_slope(np.sin(np.arange(4)))

    def test_invalid_inputs(self):
        with self.assertRaises(BaseException):
             _ = write_utils.get_slope("hello")


class TestToRanges(unittest.TestCase):

    def test_valid(self):
        actual = write_utils.to_ranges([0, 1, 2, 3, 7, 8, 9, 10])
        actual = list(actual)
        if sys.version_info.major == 3:
            expected = [range(0, 4), range(7, 11)]
            self.assertTrue(all([x == y for x, y in zip(expected, actual)]))
        else:
            expected = [xrange(0, 4), xrange(7, 11)]
            for in_x, out_x in zip(expected, actual):
                self.assertTrue(all([x == y for x, y in zip(list(in_x), list(out_x))]))


if __name__ == '__main__':
    unittest.main()
