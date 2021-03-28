# -*- coding: utf-8 -*-
"""
Created on Fri March 26 17:07:16 2021

@author: Gerd Duscher
"""
from __future__ import division, print_function, unicode_literals, \
    absolute_import
import unittest

import numpy as np
import string
import sys
sys.path.insert(0, "../../sidpy/")

import sidpy

if sys.version_info.major == 3:
    unicode = str


class TestUFunctions(unittest.TestCase):

    def test_add(self):
        input_spectrum = np.zeros([512])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset+3.
        self.assertIsInstance(new_dataset, sidpy.Dataset)
        self.assertEqual(np.array(new_dataset)[0], 3)

        input_spectrum = np.zeros([3, 3, 512])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset + 3.
        self.assertIsInstance(new_dataset, sidpy.Dataset)
        self.assertEqual(np.array(new_dataset)[0, 0, 0], 3)

    def test_sub(self):
        input_spectrum = np.zeros([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset-3.
        self.assertIsInstance(new_dataset, sidpy.Dataset)
        self.assertEqual(np.array(new_dataset)[0, 0, 0], -3)

    def test_mul(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset*3.
        self.assertIsInstance(new_dataset, sidpy.Dataset)
        self.assertEqual(np.array(new_dataset)[0, 0, 0], 3)

    def test_div(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset/3.
        self.assertIsInstance(new_dataset, sidpy.Dataset)
        self.assertEqual(np.array(new_dataset)[0, 0, 0], 1/3)

    def test_min(self):
        input_spectrum = np.zeros([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        min_dataset = dataset.min()
        self.assertIsInstance(min_dataset, float)
        self.assertEqual(min_dataset, 0)

    def test_max(self):
        input_spectrum = np.zeros([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        max_dataset = dataset.max()
        self.assertIsInstance(max_dataset, float)
        self.assertEqual(max_dataset, 0)

    def test_abs(self):
        input_spectrum = np.ones([3, 3, 3]) * -1
        dataset = sidpy.Dataset.from_array(input_spectrum)
        abs_dataset = dataset.abs()
        self.assertIsInstance(abs_dataset, sidpy.Dataset)
        self.assertEqual(abs_dataset[0, 0, 0], 1)


class TestFftFunctions(unittest.TestCase):

    def test_spectrum_fft(self):
        input_spectrum = np.zeros([512])
        x = np.mgrid[0:32] * 16
        input_spectrum[x] = 1

        dataset = sidpy.Dataset.from_array(input_spectrum)
        dataset.data_type = 'spectrum'
        dataset.units = 'counts'
        dataset.quantity = 'intensity'
        with self.assertRaises(NotImplementedError):
            dataset.fft()
        with self.assertRaises(TypeError):
            dataset.fft(dimension_type='spectral')

        dataset.set_dimension(0, sidpy.Dimension(np.arange(dataset.shape[0]) * .02, 'x'))
        dataset.x.dimension_type = 'spectral'
        dataset.x.units = 'nm'
        dataset.x.quantity = 'distance'

        fft_dataset = dataset.fft()

        self.assertEqual(np.array(fft_dataset)[0], 32+0j)

    def test_image_fft(self):
        input_image = np.zeros([512, 512])
        x, y = np.mgrid[0:32, 0:32] * 16
        input_image[x, y] = 1

        dataset = sidpy.Dataset.from_array(input_image)
        dataset.data_type = 'image'
        dataset.units = 'counts'
        dataset.quantity = 'intensity'
        with self.assertRaises(NotImplementedError):
            dataset.fft()
        with self.assertRaises(TypeError):
            dataset.fft(dimension_type='spatial')

        dataset.set_dimension(0, sidpy.Dimension(np.arange(dataset.shape[0]) * .02, 'x'))
        dataset.x.dimension_type = 'spatial'
        dataset.x.units = 'nm'
        dataset.x.quantity = 'distance'
        with self.assertRaises(TypeError):
            dataset.fft()

        dataset.set_dimension(1, sidpy.Dimension(np.arange(dataset.shape[1]) * .02, 'y'))
        dataset.y.dimension_type = 'spatial'
        dataset.y.units = 'nm'
        dataset.y.quantity = 'distance'

        fft_dataset = dataset.fft()
        self.assertEqual(np.array(fft_dataset)[0, 0], 1024+0j)

    def test_image_stack_fft(self):
        input_stack = np.zeros([3, 512, 512])
        x, y = np.mgrid[0:32, 0:32] * 16
        input_stack[:, x, y] = 1.

        dataset = sidpy.Dataset.from_array(input_stack)
        dataset.data_type = 'image_stack'
        dataset.units = 'counts'
        dataset.quantity = 'intensity'
        with self.assertRaises(NotImplementedError):
            dataset.fft()
        with self.assertRaises(TypeError):
            dataset.fft(dimension_type='spatial')

        dataset.set_dimension(0, sidpy.Dimension(np.arange(dataset.shape[0]), 'frame'))
        dataset.frame.dimension_type = 'time'
        dataset.set_dimension(1, sidpy.Dimension(np.arange(dataset.shape[1]) * .02, 'x'))
        dataset.x.dimension_type = 'spatial'
        dataset.x.units = 'nm'
        dataset.x.quantity = 'distance'
        with self.assertRaises(NotImplementedError):
            dataset.fft()

        dataset.set_dimension(2, sidpy.Dimension(np.arange(dataset.shape[2]) * .02, 'y'))
        dataset.y.dimension_type = 'spatial'
        dataset.y.units = 'nm'
        dataset.y.quantity = 'distance'

        fft_dataset = dataset.fft()

        self.assertEqual(np.array(fft_dataset)[0, 0, 0], 1024 + 0j)

    def test_spectrum_image_fft(self):
        input_si = np.zeros([3, 3, 512])
        x = np.mgrid[0:32] * 16
        input_si[:, :, x] = 1.

        dataset = sidpy.Dataset.from_array(input_si)
        dataset.data_type = 'spectral_image'
        dataset.units = 'counts'
        dataset.quantity = 'intensity'
        with self.assertRaises(NotImplementedError):
            dataset.fft()
        with self.assertRaises(TypeError):
            dataset.fft(dimension_type='spectral')

        dataset.set_dimension(0, sidpy.Dimension(np.arange(dataset.shape[0]) * .02, 'x'))
        dataset.x.dimension_type = 'spatial'
        dataset.x.units = 'nm'
        dataset.x.quantity = 'distance'
        with self.assertRaises(NotImplementedError):
            dataset.fft()
        dataset.set_dimension(1, sidpy.Dimension(np.arange(dataset.shape[1]) * .02, 'y'))
        dataset.y.dimension_type = 'spatial'
        dataset.y.units = 'nm'
        dataset.y.quantity = 'distance'

        dataset.set_dimension(2, sidpy.Dimension(np.arange(dataset.shape[0]) * .02, 'spec'))
        dataset.spec.dimension_type = 'spectral'
        dataset.spec.units = 'i'
        dataset.spec.quantity = 'energy'

        fft_dataset = dataset.fft()

        self.assertEqual(np.array(fft_dataset)[0, 0, 0], 32+0j)


if __name__ == '__main__':
    unittest.main()
