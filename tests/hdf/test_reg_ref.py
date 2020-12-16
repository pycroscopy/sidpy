# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 15:07:16 2018

@author: Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, \
    absolute_import
import unittest
import os
import sys
import h5py
import numpy as np

from . import data_utils

sys.path.append("../../sidpy/")
from sidpy.hdf.hdf_utils import get_attr
from sidpy.hdf import reg_ref

if sys.version_info.major == 3:
    unicode = str


class TestRegRef(unittest.TestCase):

    def setUp(self):
        data_utils.make_beps_file()

    def tearDown(self):
        data_utils.delete_existing_file(data_utils.std_beps_path)


class TesetGetIndicesForRegionRef(TestRegRef):

    def test_corners(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            ref_in = get_attr(h5_main, 'even_rows')
            ret_val = reg_ref.get_indices_for_region_ref(h5_main, ref_in,
                                                         'corners')
            expected_pos = np.repeat(np.arange(h5_main.shape[0])[::2], 2)
            expected_spec = np.tile(np.array([0, h5_main.shape[1] - 1]),
                                    expected_pos.size // 2)
            expected_corners = np.vstack((expected_pos, expected_spec)).T
            self.assertTrue(np.allclose(ret_val, expected_corners))

    def test_slices(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            ref_in = get_attr(h5_main, 'even_rows')
            ret_val = reg_ref.get_indices_for_region_ref(h5_main, ref_in,
                                                         'slices')
            spec_slice = slice(0, h5_main.shape[1] - 1, None)
            expected_slices = np.array(
                [[slice(x, x, None), spec_slice] for x in
                 np.arange(h5_main.shape[0])[::2]])
            self.assertTrue(np.all(ret_val == expected_slices))


class TestWriteRegRef(TestRegRef):

    def test_main_one_dim(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        data = np.random.rand(7)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=data)
            reg_refs = {'even_rows': (slice(0, None, 2)),
                        'odd_rows': (slice(1, None, 2))}
            reg_ref.write_region_references(h5_dset, reg_refs,
                                            add_labels_attr=True)
            self.assertEqual(len(h5_dset.attrs), 1 + len(reg_refs))
            actual = get_attr(h5_dset, 'labels')
            self.assertTrue(np.all(
                [x == y for x, y in zip(actual, ['even_rows', 'odd_rows'])]))

            expected_data = [data[0:None:2], data[1:None:2]]
            written_data = [h5_dset[h5_dset.attrs['even_rows']],
                            h5_dset[h5_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)

    def test_main_1st_dim(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        data = np.random.rand(5, 7)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=data)
            reg_refs = {'even_rows': (slice(0, None, 2), slice(None)),
                        'odd_rows': (slice(1, None, 2), slice(None))}
            reg_ref.write_region_references(h5_dset, reg_refs,
                                            add_labels_attr=True)
            self.assertEqual(len(h5_dset.attrs), 1 + len(reg_refs))
            actual = get_attr(h5_dset, 'labels')
            self.assertTrue(np.all(
                [x == y for x, y in zip(actual, ['even_rows', 'odd_rows'])]))

            expected_data = [data[0:None:2], data[1:None:2]]
            written_data = [h5_dset[h5_dset.attrs['even_rows']],
                            h5_dset[h5_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)

    def test_main_2nd_dim(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        data = np.random.rand(5, 7)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=data)
            reg_refs = {'even_rows': (slice(None), slice(0, None, 2)),
                        'odd_rows': (slice(None), slice(1, None, 2))}
            reg_ref.write_region_references(h5_dset, reg_refs,
                                            add_labels_attr=False)
            self.assertEqual(len(h5_dset.attrs), len(reg_refs))
            self.assertTrue('labels' not in h5_dset.attrs.keys())

            expected_data = [data[:, 0:None:2], data[:, 1:None:2]]
            written_data = [h5_dset[h5_dset.attrs['even_rows']],
                            h5_dset[h5_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)


class TestSimpleRegionRefCopy(TestRegRef):

    def test_base(self):
        # based on test_hdf_writer.test_write_legal_reg_ref_multi_dim_data()
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            data = np.random.rand(5, 7)
            h5_orig_dset = h5_f.create_dataset('test', data=data)
            self.assertIsInstance(h5_orig_dset, h5py.Dataset)

            attrs = {'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                'odd_rows': (slice(1, None, 2), slice(None))}}

            data_utils.write_main_reg_refs(h5_orig_dset, attrs['labels'])
            h5_f.flush()

            # two atts point to region references. one for labels
            self.assertEqual(len(h5_orig_dset.attrs), 1 + len(attrs['labels']))

            # check if the labels attribute was written:

            self.assertTrue(np.all([x in list(attrs['labels'].keys()) for x in
                                    get_attr(h5_orig_dset,
                                             'labels')]))

            expected_data = [data[:None:2], data[1:None:2]]
            written_data = [h5_orig_dset[h5_orig_dset.attrs['even_rows']],
                            h5_orig_dset[h5_orig_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

            # Now write a new dataset without the region reference:
            h5_new_dset = h5_f.create_dataset('other', data=data)
            self.assertIsInstance(h5_orig_dset, h5py.Dataset)
            h5_f.flush()

            for key in attrs['labels'].keys():
                reg_ref.simple_region_ref_copy(h5_orig_dset, h5_new_dset, key)

            # now check to make sure that this dataset also has the same region references:
            written_data = [h5_new_dset[h5_new_dset.attrs['even_rows']],
                            h5_new_dset[h5_new_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)


class TestCreateRegionRef(TestRegRef):

    def test_base(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        data = np.random.rand(5, 7)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Source', data=data)
            pos_inds = np.arange(0, h5_dset.shape[0], 2)
            ref_inds = [((pos_start, 0), (pos_start, h5_dset.shape[1] - 1)) for
                        pos_start in pos_inds]
            ref_inds = np.array(ref_inds)
            this_reg_ref = reg_ref.create_region_reference(h5_dset, ref_inds)
            ref_slices = list()
            for start, stop in ref_inds:
                ref_slices.append(
                    [slice(start[0], stop[0] + 1), slice(start[1], None)])

            h5_reg = h5_dset[this_reg_ref]

            h5_slice = np.vstack(
                [h5_dset[pos_slice, spec_slice] for (pos_slice, spec_slice) in
                 ref_slices])

            self.assertTrue(np.allclose(h5_reg, h5_slice))

        os.remove(file_path)


class TestGetRegion(TestRegRef):

    def test_illegal_01(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            with self.assertRaises(KeyError):
                reg_ref.get_region(h5_f['/Raw_Measurement/source_main'],
                                   'non_existent')

    def test_legal_01(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_source = h5_f['/Raw_Measurement/source_main']
            returned = reg_ref.get_region(h5_source, 'even_rows')
            expected = h5_source[slice(0, h5_source.shape[0], 2)]
            self.assertTrue(np.allclose(returned, expected))


class TestCleanRegRef(TestRegRef):

    def test_1d(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.random.rand(7))
            ref_in = (slice(0, None, 2))
            cleaned = reg_ref.clean_reg_ref(h5_dset, ref_in)
            self.assertEqual(ref_in, cleaned[0])
        os.remove(file_path)

    def test_2d(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.random.rand(7, 5))
            ref_in = (slice(0, None, 2), slice(None))
            cleaned = reg_ref.clean_reg_ref(h5_dset, ref_in)
            self.assertTrue(np.all([x == y for x, y in zip(ref_in, cleaned)]))
        os.remove(file_path)

    def test_illegal_too_many_slices(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.random.rand(7, 5))
            ref_in = (slice(0, None, 2), slice(None), slice(1, None, 2))
            with self.assertRaises(ValueError):
                _ = reg_ref.clean_reg_ref(h5_dset, ref_in)

        os.remove(file_path)

    def test_illegal_too_few_slices(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.random.rand(7, 5))
            ref_in = (slice(0, None, 2))
            with self.assertRaises(ValueError):
                _ = reg_ref.clean_reg_ref(h5_dset, ref_in)

        os.remove(file_path)

    def test_out_of_bounds(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.random.rand(7, 5))
            ref_in = (slice(0, 13, 2), slice(None))
            expected = (slice(0, 7, 2), slice(None))
            cleaned = reg_ref.clean_reg_ref(h5_dset, ref_in, verbose=False)
            self.assertTrue(
                np.all([x == y for x, y in zip(expected, cleaned)]))
        os.remove(file_path)


class TestAttemptRegRefBuild(TestRegRef):

    def test_spec(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Indices', data=np.random.rand(2, 5))
            dim_names = ['Bias', 'Cycle']
            expected = {'Bias': (slice(0, 1), slice(None)),
                        'Cycle': (slice(1, 2), slice(None))}
            if sys.version_info.major == 3:
                with self.assertWarns(UserWarning):
                    cleaned = reg_ref.attempt_reg_ref_build(h5_dset, dim_names)
            else:
                cleaned = reg_ref.attempt_reg_ref_build(h5_dset, dim_names)
            for key, value in expected.items():
                self.assertEqual(value, cleaned[key])
        os.remove(file_path)

    def test_pos(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Indices', data=np.random.rand(5, 2))
            dim_names = ['Bias', 'Cycle']
            expected = {'Bias': (slice(None), slice(0, 1)),
                        'Cycle': (slice(None), slice(1, 2))}
            if sys.version_info.major == 3:
                with self.assertWarns(UserWarning):
                    cleaned = reg_ref.attempt_reg_ref_build(h5_dset, dim_names)
            else:
                cleaned = reg_ref.attempt_reg_ref_build(h5_dset, dim_names)
            for key, value in expected.items():
                self.assertEqual(value, cleaned[key])
        os.remove(file_path)

    def test_pos_too_many_dims(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Indices', data=np.random.rand(5, 2))
            dim_names = ['Bias', 'Cycle', 'Blah']
            ret_val = reg_ref.attempt_reg_ref_build(h5_dset, dim_names)
            self.assertEqual(ret_val, dict())
        os.remove(file_path)

    def test_pos_too_few_dims(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Indices', data=np.random.rand(5, 2))
            dim_names = ['Bias']
            ret_val = reg_ref.attempt_reg_ref_build(h5_dset, dim_names)
            self.assertEqual(ret_val, dict())
        os.remove(file_path)
