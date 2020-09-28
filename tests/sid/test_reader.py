# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, \
    absolute_import
import unittest
import sys
import os
sys.path.append("../../sidpy/")
from sidpy.sid import Reader


class TestCanReadNotImplemented(unittest.TestCase):

    def setUp(self):

        class DummyReader(Reader):

            def read(self):
                pass

        self.reader_class = DummyReader

    def test_bad_init_obj(self):
        with self.assertRaises(TypeError):
            _ = self.reader_class(['haha.txt'])

    def test_file_does_not_exist(self):
        with self.assertRaises(FileNotFoundError):
            _ = self.reader_class('haha.txt')

    def test_valid_file(self):

        file_path = os.path.abspath('blah.txt')
        with open(file_path, mode='w') as file_handle:
            file_handle.write('Nothing')

        reader = self.reader_class(file_path)
        with self.assertRaises(NotImplementedError):
            reader.can_read()

        os.remove(file_path)

    def test_invalid_file_object(self):
        with self.assertRaises(TypeError):
            _ = self.reader_class(1.23234)

        with self.assertRaises(TypeError):
            _ = self.reader_class({'fdfdfd': 33})

        with self.assertRaises(TypeError):
            _ = self.reader_class([1, '2343434', False])

    def test_empty_file_path(self):
        with self.assertRaises(ValueError):
            _ = self.reader_class('    ')

    def test_space_in_file_path(self):
        file_path = './blah.txt'
        with open(file_path, mode='w') as file_handle:
            file_handle.write('Nothing')

        reader = self.reader_class('    ' + file_path + '    ')

        self.assertEqual(reader._input_file_path, file_path)

        os.remove(file_path)


class TestCanReadExtVariations(unittest.TestCase):

    def test_invalid_ext_obj_type(self):
        class DummyReader(Reader):
            def can_read(self):
                with self.assertRaises(TypeError):
                    return super(DummyReader, self).can_read(extension={'txt': 5})

    def base_test_extension_variations(self, exts, file_paths):
        class DummyReader(Reader):
            def can_read(self):
                return super(DummyReader, self).can_read(extension=exts)

        if isinstance(file_paths, str):
            file_paths = [file_paths]

        for file_path in file_paths:
            file_path = os.path.abspath(file_path)

            with open(file_path, mode='w') as file_handle:
                file_handle.write('Nothing')

            reader = DummyReader(file_path)

            self.assertTrue(reader.can_read())

            os.remove(file_path)

    def test_upper_case_extension(self):
        self.base_test_extension_variations('TXT', ['blah.TXT', 'blah.txt'])

    def test_lower_case_extension(self):
        self.base_test_extension_variations('txt', ['blah.TXT', 'blah.txt'])

    def test_multi_cases_extensions(self):
        self.base_test_extension_variations(['txt', 'RST'],
                                            ['blah.TXT', 'blah.rst'])

    def test_dot_before_extensions(self):
        self.base_test_extension_variations('.txt', ['blah.TXT', 'blah.txt'])


class TestCanReadOneExt(unittest.TestCase):

    def setUp(self):

        class DummyReader(Reader):

            def can_read(self):
                return super(DummyReader, self).can_read(extension='txt')

        self.reader_class = DummyReader

    def base_test(self, file_path, assert_func):

        file_path = os.path.abspath(file_path)

        with open(file_path, mode='w') as file_handle:
                file_handle.write('Nothing')

        reader = self.reader_class(file_path)

        assert_func(reader.can_read())

        os.remove(file_path)

    def test_valid_extension(self):
        self.base_test('blah.txt', self.assertTrue)

    def test_invalid_extension(self):
        self.base_test('blah.rst', self.assertFalse)

    def test_invalid_dir(self):
        reader = self.reader_class('.')
        self.assertFalse(reader.can_read())


class TestCanReadMultiExts(unittest.TestCase):

    def setUp(self):
        class DummyReader(Reader):

            def can_read(self):
                exts = ['txt', 'rst']
                return super(DummyReader, self).can_read(extension=exts)

            def read(self):
                pass

        self.reader_class = DummyReader

    def base_test(self, file_path, assert_func):
        file_path = os.path.abspath(file_path)

        with open(file_path, mode='w') as file_handle:
            file_handle.write('Nothing')

        reader = self.reader_class(file_path)

        assert_func(reader.can_read())

        os.remove(file_path)

    def test_valid_txt_file(self):
        self.base_test('blah.txt', self.assertTrue)

    def test_valid_rst_file(self):
        self.base_test('blah.rst', self.assertTrue)

    def test_invalid_file(self):
        self.base_test('blah.png', self.assertFalse)

    def test_case_insensitive(self):
        self.base_test('ALL_CAPS.RST', self.assertTrue)


if __name__ == '__main__':
    unittest.main()
