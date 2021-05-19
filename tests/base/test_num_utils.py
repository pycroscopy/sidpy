# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath, Rama Vasudevan
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys
import numpy as np

sys.path.append("../../sidpy/")
from sidpy.base.num_utils import *

if sys.version_info.major == 3:
    unicode = str
    xrange = range
    

class TestGetSlope(unittest.TestCase):

    def test_linear(self):
        expected = 0.25
        actual = get_slope(np.arange(-1, 1, expected))
        self.assertEqual(expected, actual)

    def test_linear_dirty(self):
        # When reading from HDF5, rounding errors can result in minor variations in the diff
        expected = 0.25E-9
        vector = np.arange(-1E-9, 1E-9, expected)
        round_error = np.random.rand(vector.size) * 1E-14
        vector += round_error
        actual = get_slope(vector, tol=1E-3)
        self.assertAlmostEqual(expected, actual)

    def test_invalid_tolerance(self):
        with self.assertRaises(TypeError):
            _ = get_slope(np.sin(np.arange(4)), tol="hello")

    def test_non_linear(self):
        with self.assertRaises(ValueError):
            _ = get_slope(np.sin(np.arange(4)))

    def test_invalid_inputs(self):
        with self.assertRaises(BaseException):
             _ = get_slope("hello")


class TestToRanges(unittest.TestCase):

    def test_valid(self):
        actual = to_ranges([0, 1, 2, 3, 7, 8, 9, 10])
        actual = list(actual)
        if sys.version_info.major == 3:
            expected = [range(0, 4), range(7, 11)]
            self.assertTrue(all([x == y for x, y in zip(expected, actual)]))
        else:
            expected = [xrange(0, 4), xrange(7, 11)]
            for in_x, out_x in zip(expected, actual):
                self.assertTrue(all([x == y for x, y in zip(list(in_x), list(out_x))]))


class TestContainsIntegers(unittest.TestCase):

    def test_typical(self):
        self.assertTrue(contains_integers([1, 2, -3, 4]))
        self.assertTrue(contains_integers(range(5)))
        self.assertTrue(
            contains_integers([2, 5, 8, 3], min_val=2))
        self.assertTrue(contains_integers(np.arange(5)))
        self.assertFalse(
            contains_integers(np.arange(5), min_val=2))
        self.assertFalse(
            contains_integers([1, 4.5, 2.2, -1]))
        self.assertFalse(
            contains_integers([1, -2, 5], min_val=1))
        self.assertFalse(
            contains_integers(['dsss', 34, 1.23, None]))
        self.assertFalse(contains_integers([]))

        with self.assertRaises(TypeError):
            _ = contains_integers(None)
        with self.assertRaises(TypeError):
            _ = contains_integers(14)

    def test_illegal_min_val(self):
        with self.assertRaises(TypeError):
            _ = contains_integers([1, 2, 3, 4],
                                                     min_val='hello')

        with self.assertRaises(TypeError):
            _ = contains_integers([1, 2, 3, 4],
                                                     min_val=[1, 2])

        with self.assertRaises(ValueError):
            _ = contains_integers([1, 2, 3, 4],
                                                     min_val=1.234)


class TestIntegersToSlices(unittest.TestCase):

    def test_illegal_inputs(self):
        with self.assertRaises(TypeError):
            integers_to_slices(slice(1, 15))
        with self.assertRaises(ValueError):
            integers_to_slices(
                [-1.43, 34.6565, 45.344, 5 + 6j])
        with self.assertRaises(ValueError):
            integers_to_slices(
                ['asdds', None, True, 45.344, 5 + 6j])

    def test_positive(self):
        expected = [slice(0, 3), slice(7, 8), slice(14, 18), slice(22, 23),
                    slice(27, 28), slice(29, 30), slice(31, 32)]
        inputs = np.hstack([range(item.start, item.stop) for item in expected])
        ret_val = integers_to_slices(inputs)
        self.assertEqual(expected, ret_val)

    def test_negative(self):
        expected = [slice(-7, -4), slice(-2, 3), slice(14, 18), slice(22, 23),
                    slice(27, 28), slice(29, 30)]
        inputs = np.hstack([range(item.start, item.stop) for item in expected])
        ret_val = integers_to_slices(inputs)
        self.assertEqual(expected, ret_val)
        
        
class TestGetExponent(unittest.TestCase):

    def test_negative_small(self):
        expected = -7
        self.assertEqual(expected,
                         get_exponent(np.arange(5) * -10 ** expected))

    def test_positive_large(self):
        expected = 4
        self.assertEqual(expected,
                         get_exponent(np.arange(6) * 10 ** expected))

    def test_mixed_large(self):
        expected = 4
        self.assertEqual(expected,
                         get_exponent(np.random.randint(-8, high=3, size=(5, 5)) * 10 ** expected))

    def test_illegal_type(self):
        with self.assertRaises(TypeError):
            _ = get_exponent('hello')
            _ = get_exponent([1, 2, 3])
            _ = get_exponent([0, 1, np.nan])
                
class TestBuildIndValMatrices(unittest.TestCase):
    '''Testing the build_ind_val_matrices function'''
    def test_not_list_or_tuple(self):
        with self.assertRaises(TypeError):
            #try putting in a dictionary
            unit_values = {'values':(0,1,2)}
            _,_ = build_ind_val_matrices (unit_values)

            #try a numpy array
            unit_values = np.array([0,1,2,3])
            _, _ = build_ind_val_matrices(unit_values)

    def test_not_1D(self):
        with self.assertRaises(ValueError):
            # try a 2D matrix
            unit_values = [np.random.normal(loc=1,scale=1,size=(5,5))]
            _, _ = build_ind_val_matrices(unit_values)

    def test_standard_case(self):
        #here we want to assert that a standard case works
        #two spectroscopic dimensions - [[0,1], [10,20]]
        unit_values = [[0,1], [10,20]]
        ind_mat, val_mat = build_ind_val_matrices(unit_values)
        ind_mat_true = np.array([[0,0],[1,0], [0,1],[1,1]])
        val_mat_true =  np.array([[0., 10.], [1., 10.],
                                  [0., 20.], [1., 20.]])
        self.assertTrue(np.isclose(ind_mat, ind_mat_true).all() == True)
        self.assertTrue(np.isclose(val_mat, val_mat_true).all() ==True)


if __name__ == '__main__':
    unittest.main()