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
        with self.assertWarns(DeprecationWarning):
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





if __name__ == '__main__':
    unittest.main()
