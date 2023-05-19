# -*- coding: utf-8 -*-
"""
Created on Fri March 26 17:07:16 2021

@author: Gerd Duscher
"""
from __future__ import division, print_function, unicode_literals, \
    absolute_import
import unittest

import numpy as np
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

        new_dataset.compute()
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
        self.assertEqual(abs_dataset[0, 0, 0].compute(), 1)
        new_dataset = dataset.__abs__()
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_angle(self):
        input_spectrum = np.ones([3, 3, 3]) * -1
        dataset = sidpy.Dataset.from_array(input_spectrum)
        angle_dataset = dataset.angle()
        self.assertIsInstance(angle_dataset, sidpy.Dataset)
        self.assertEqual(float(angle_dataset[0, 0, 0]), np.pi)

    def test_dot(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.dot(dataset)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_t(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.T
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_transpose(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.transpose()
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_ravel(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.ravel()
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_choose(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.choose([2, 3, 1, 0])
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_radd(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset+3.
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_and(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.__and__(dataset)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_rand(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.__rand__(dataset)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_div(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset/dataset
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_rdiv(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset/np.array(dataset)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_gt(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset > 3
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_ge(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset >= 3
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_invert(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = ~dataset
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_lshift(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.__lshift__(1)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_lt(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset < 3
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_le(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset <= 3
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_mod(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset % 3
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_rmod(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.__rmod__(2)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_rmul(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.__rmul__(2)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_ne(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset != 2
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_neg(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.__neg__()
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_or(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.__or__(dataset)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_ror(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.__ror__(dataset)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_pos(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.__pos__()
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_pow(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset**2
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_rpow(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.__rpow__(2)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_rshift(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.__rshift__(2)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_rrshift(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.__rrshift__(2)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_rsub(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.__rsub__(2)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_rtruediv(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.__rtruediv__(2)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_rfloordiv(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.__rfloordiv__(2)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_xor(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.__xor__(2)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_rxor(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.__rxor__(2)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_matmul(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset @ dataset
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_rmatmul(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.__rmatmul__(dataset)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_divmod(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset, _ = dataset.__divmod__(2)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_rdivmod(self):
        input_spectrum = np.ones([3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset, _ = dataset.__rdivmod__(8)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_real(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.real
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_imag(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.imag
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_conj(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.conj()
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_clip(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.clip()
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_sum(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.sum(axis=1)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_mean(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.mean(axis=1)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_squeeze(self):
        input_spectrum = np.ones([3, 1, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.squeeze()
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_swapaxes(self):
        input_spectrum = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_spectrum)
        new_dataset = dataset.swapaxes(0, 1)
        self.assertIsInstance(new_dataset, sidpy.Dataset)

    def test_ufunc(self):
        # Todo: More testing for better coverage
        input_image = np.ones([3, 3, 3])
        dataset = sidpy.Dataset.from_array(input_image)
        new_dataset = np.sin(dataset)
        self.assertIsInstance(new_dataset, sidpy.Dataset)
        new_dataset = dataset @ dataset
        self.assertIsInstance(new_dataset, sidpy.Dataset)


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

        dataset.set_dimension(2, sidpy.Dimension(np.arange(dataset.shape[2]) * .02, 'spec'))
        dataset.spec.dimension_type = 'spectral'
        dataset.spec.units = 'i'
        dataset.spec.quantity = 'energy'

        fft_dataset = dataset.fft()

        self.assertEqual(np.array(fft_dataset)[0, 0, 0], 32+0j)


if __name__ == '__main__':
    unittest.main()
