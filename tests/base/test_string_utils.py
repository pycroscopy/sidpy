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

    def test_string_unicode_handling(self):
        expected = ['a', 'bc', 'µm']
        returned = clean_string_att(expected)
        for ind,x in enumerate(expected): 
            assert returned[ind].decode('utf-8') == expected[ind]
        
            

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


class TestStrToOther(unittest.TestCase):

    def test_invalid_input_obj_type(self):
        for val in [1.23, {'1we': 123}, ['dssd'], True, None]:
            with self.assertRaises(TypeError):
                str_to_other(val)

    def base_test(self, inputs, out_type):
        for val in inputs:
            ret = str_to_other(str(val))
            self.assertEqual(val, ret)
            self.assertIsInstance(ret, out_type)

    def test_int(self):
        self.base_test([23, -235457842], int)

    def test_float(self):
        self.base_test([23.45643, -2354.57842], float)

    def test_exp(self):
        self.base_test([3.14E3, -4.3E-5], float)

    def test_str(self):
        self.base_test(['hello', '1fd353'], str)

    def test_bool(self):
        for val in ['true', 'TRUE', 'True']:
            ret = str_to_other(val)
            self.assertEqual(ret, True)
            self.assertIsInstance(ret, bool)
        for val in ['false', 'FALSE', 'False']:
            ret = str_to_other(val)
            self.assertEqual(ret, False)
            self.assertIsInstance(ret, bool)


class TestRemoveExtraDelimiters(unittest.TestCase):

    def test_invalid_sep_type(self):
        for sep in [14, {'fdfd': 45}, [' ', ', '], True, (23, None)]:
            with self.assertRaises(TypeError):
                remove_extra_delimiters('fddfdf dfref', separator=sep)

    def test_invalid_line_type(self):
        for line in [14, {'fdfd': 45}, [' ', ', '], True, (23, None)]:
            with self.assertRaises(TypeError):
                remove_extra_delimiters(line, separator='-')

    def test_empty_delim(self):
        with self.assertRaises(ValueError):
            remove_extra_delimiters('this is a test', '')

    def typical_case(self, pad=False):
        words = ['this', 'is', 'a', 'test']
        for sep in [' ', '-']:
            line = sep.join(words)
            if pad:
                dirty = sep * 4 + line + sep * 3
            else:
                dirty = line
            clean = remove_extra_delimiters(dirty, separator=sep)
            self.assertEqual(line, clean)
            self.assertIsInstance(clean, str)

    def test_single_delim(self):
        self.typical_case(pad=False)

    def test_delims_before_or_after(self):
        self.typical_case(pad=True)

    def test_multiple_consecutive_delims(self):
        line = 'this    is a   test  sentence'
        words = ['this', 'is', 'a', 'test', 'sentence']
        clean = remove_extra_delimiters(line, separator=' ')
        self.assertEqual(clean, ' '.join(words))
        line = 'this====is=a==test=========sentence'
        clean = remove_extra_delimiters(line, separator='=')
        self.assertEqual(clean, '='.join(words))


if __name__ == '__main__':
    unittest.main()
