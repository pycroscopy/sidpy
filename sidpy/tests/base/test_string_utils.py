# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys

sys.path.append("../../sidpy/")
from sidpy.base.string_utils import *

if sys.version_info.major == 3:
    unicode = str


class TestCleanStringAtt(unittest.TestCase):

    def test_float(self):
        expected = 5.321
        self.assertEqual(expected, clean_string_att(expected))

    def test_str(self):
        expected = 'test'
        self.assertEqual(expected, clean_string_att(expected))

    def test_num_array(self):
        expected = [1, 2, 3.456]
        self.assertEqual(expected, clean_string_att(expected))

    def test_str_list(self):
        expected = ['a', 'bc', 'def']
        returned = clean_string_att(expected)
        expected = np.array(expected, dtype='S')
        for exp, act in zip(expected, returned):
            self.assertEqual(exp, act)

    def test_str_tuple(self):
        expected = ('a', 'bc', 'def')
        returned = clean_string_att(expected)
        expected = np.array(expected, dtype='S')
        for exp, act in zip(expected, returned):
            self.assertEqual(exp, act)
            

class TestFormattedStrToNum(unittest.TestCase):

    def test_typical(self):
        self.assertEqual(
            formatted_str_to_number("4.32 MHz", ["MHz", "kHz"], [1E+6, 1E+3]), 4.32E+6)

    def test_wrong_types(self):
        with self.assertRaises(TypeError):
            _ = formatted_str_to_number("4.32 MHz", ["MHz", "kHz"],
                                                              [1E+6, 1E+3], separator=14)
        with self.assertRaises(TypeError):
            _ = formatted_str_to_number({'dfdfd': 123}, ["MHz"], [1E+6])
        with self.assertRaises(TypeError):
            _ = formatted_str_to_number("dfdfdf", ["MHz"], 1E+6)
        with self.assertRaises(TypeError):
            _ = formatted_str_to_number("jkjk", ["MHz", 1234], [1E+6, 1E+4])
        with self.assertRaises(TypeError):
            _ = formatted_str_to_number("4.32 MHz", ["MHz", "kHz"], [{'dfdfd': 13}, 1E+3])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            _ = formatted_str_to_number("4.32 MHz", ["MHz"], [1E+6, 1E+3])
        with self.assertRaises(ValueError):
            _ = formatted_str_to_number("4.32 MHz", ["MHz", "kHz"], [1E+3])
        with self.assertRaises(ValueError):
            _ = formatted_str_to_number("4.32-MHz", ["MHz", "kHz"], [1E+6, 1E+3])
        with self.assertRaises(ValueError):
            _ = formatted_str_to_number("haha MHz", ["MHz", "kHz"], [1E+6, 1E+3])
        with self.assertRaises(ValueError):
            _ = formatted_str_to_number("1.2.3.4 MHz", ["MHz", "kHz"], [1E+6, 1E+3])
        with self.assertRaises(ValueError):
            _ = formatted_str_to_number("MHz", ["MHz", "kHz"], [1E+6, 1E+3])


class TestFormatQuantity(unittest.TestCase):

    def test_typical(self):
        qty_names = ['sec', 'mins', 'hours', 'days']
        qty_factors = [1, 60, 3600, 3600*24]
        ret_val = format_quantity(315, qty_names, qty_factors)
        self.assertEqual(ret_val, '5.25 mins')
        ret_val = format_quantity(6300, qty_names, qty_factors)
        self.assertEqual(ret_val, '1.75 hours')

    def test_unequal_lengths(self):
        with self.assertRaises(ValueError):
            _ = format_quantity(315, ['sec', 'mins', 'hours'], [1, 60, 3600, 3600 * 24])
        with self.assertRaises(ValueError):
            _ = format_quantity(315, ['sec', 'mins', 'hours'], [1, 60])

    def test_incorrect_element_types(self):
        with self.assertRaises(TypeError):
            _ = format_quantity(315, ['sec', 14, 'hours'], [1, 60, 3600 * 24])

    def test_incorrect_number_to_format(self):
        with self.assertRaises(TypeError):
            _ = format_quantity('hello', ['sec', 'mins', 'hours'], [1, 60, 3600])

    def test_not_iterable(self):
        with self.assertRaises(TypeError):
            _ = format_quantity(315, 14, [1, 60, 3600])

        with self.assertRaises(TypeError):
            _ = format_quantity(315, ['sec', 'mins', 'hours'], slice(None))


class TestTimeSizeFormatting(unittest.TestCase):

    def test_format_time(self):
        ret_val = format_time(315)
        self.assertEqual(ret_val, '5.25 mins')
        ret_val = format_time(6300)
        self.assertEqual(ret_val, '1.75 hours')

    def test_format_size(self):
        ret_val = format_size(15.23)
        self.assertEqual(ret_val, '15.23 bytes')
        ret_val = format_size(5830418104.32)
        self.assertEqual(ret_val, '5.43 GB')
        

class TestValidateStringArgs(unittest.TestCase):

    def test_empty(self):
        with self.assertRaises(ValueError):
            _ = validate_string_args(['      '], ['meh'])

    def test_spaces(self):
        expected = 'fd'
        [ret] = validate_string_args(['  ' + expected + '    '], ['meh'])
        self.assertEqual(expected, ret)

    def test_single(self):
        expected = 'fd'
        [ret] = validate_string_args(expected, 'meh')
        self.assertEqual(expected, ret)

    def test_multi(self):
        expected = ['abc', 'def']
        returned = validate_string_args(['   ' + expected[0], expected[1] + '    '], ['meh', 'foo'])
        for exp, ret in zip(expected, returned):
            self.assertEqual(exp, ret)

    def test_not_string_lists(self):
        with self.assertRaises(TypeError):
            _ = validate_string_args([14], ['meh'])

        with self.assertRaises(TypeError):
            _ = validate_string_args(14, ['meh'])

        with self.assertRaises(TypeError):
            _ = validate_string_args({'dfdf': 14}, ['meh'])

    def test_name_not_string(self):
        actual = ['ghghg']
        ret = validate_string_args(actual, [np.arange(3)])
        self.assertEqual(ret, actual)

    def test_unequal_lengths(self):
        expected = ['a', 'b']
        actual = validate_string_args(expected + ['c'], ['a', 'b'])
        for exp, ret in zip(expected, actual):
            self.assertEqual(exp, ret)

    def test_names_not_list_of_strings(self):
        with self.assertRaises(TypeError):
            _ = validate_string_args(['a', 'v'], {'a': 1, 'v': 43})
            

if __name__ == '__main__':
    unittest.main()
