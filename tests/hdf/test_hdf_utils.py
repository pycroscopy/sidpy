# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, \
    absolute_import
import unittest
import tempfile
import os
import sys
import tempfile
from numbers import Number
import h5py
import numpy as np
import dask.array as da
from enum import Enum
sys.path.append("../../sidpy/")
from sidpy.base.dict_utils import flatten_dict
from sidpy.hdf import hdf_utils


from . import data_utils

if sys.version_info.major == 3:
    unicode = str


class TestLazyLoadArray(unittest.TestCase):

    def test_dask_input(self):
        arr = da.random.random(2, 3)
        self.assertTrue(np.allclose(arr, hdf_utils.lazy_load_array(arr)))

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            _ = hdf_utils.lazy_load_array([1, 2, 3])

    def test_numpy(self):
        np_arr = np.random.rand(2, 3)
        da_arr = da.from_array(np_arr, chunks=np_arr.shape)
        self.assertTrue(np.allclose(da_arr, hdf_utils.lazy_load_array(np_arr)))

    def test_h5_dset_no_chunks(self):
        h5_path = 'blah.h5'
        data_utils.delete_existing_file(h5_path)
        with h5py.File(h5_path, mode='w') as h5_f:
            np_arr = np.random.rand(2, 3)
            h5_dset = h5_f.create_dataset('Test', data=np_arr)
            da_arr = da.from_array(np_arr, chunks=np_arr.shape)
            self.assertTrue(np.allclose(da_arr, hdf_utils.lazy_load_array(h5_dset)))
        os.remove(h5_path)

    def test_h5_dset_w_chunks(self):
        h5_path = 'blah.h5'
        data_utils.delete_existing_file(h5_path)
        with h5py.File(h5_path, mode='w') as h5_f:
            np_arr = np.random.rand(200, 30)
            h5_dset = h5_f.create_dataset('Test', data=np_arr, chunks=(1, 30))
            da_arr = da.from_array(np_arr, chunks=h5_dset.chunks)
            self.assertTrue(np.allclose(da_arr, hdf_utils.lazy_load_array(h5_dset)))
        os.remove(h5_path)


class TestHDFUtilsBase(unittest.TestCase):

    def setUp(self):
        data_utils.make_beps_file()

    def tearDown(self):
        data_utils.delete_existing_file(data_utils.std_beps_path)


class TestGetAttr(TestHDFUtilsBase):

    def test_not_hdf_dset(self):
        with self.assertRaises(TypeError):
            hdf_utils.get_attr(np.arange(3), 'units')

    def test_illegal_attr_type(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            with self.assertRaises(TypeError):
                hdf_utils.get_attr(h5_f['/Raw_Measurement/source_main'], 14)

    def test_illegal_multiple_attrs(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            with self.assertRaises(TypeError):
                hdf_utils.get_attr(h5_f['/Raw_Measurement/source_main'], ['quantity', 'units'])

    def test_illegal_non_existent_attr(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            with self.assertRaises(KeyError):
                hdf_utils.get_attr(h5_f['/Raw_Measurement/source_main'], 'non_existent')

    def test_legal_01(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            returned = hdf_utils.get_attr(h5_f['/Raw_Measurement/source_main'], 'units')
            self.assertEqual(returned, 'A')

    def test_legal_02(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            returned = hdf_utils.get_attr(h5_f['/Raw_Measurement/Position_Indices'], 'labels')
            self.assertTrue(np.all(returned == ['X', 'Y']))

    def test_legal_03(self):
        attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                 'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
            for key, expected_value in attrs.items():
                self.assertTrue(np.all(hdf_utils.get_attr(h5_group, key) == expected_value))


class TestGetAttributes(TestHDFUtilsBase):

    def test_one(self):
        attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                 'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
        sub_attrs = ['att_3']
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
            returned_attrs = hdf_utils.get_attributes(h5_group, sub_attrs[-1])
            self.assertIsInstance(returned_attrs, dict)
            for key in sub_attrs:
                self.assertTrue(np.all(returned_attrs[key] == attrs[key]))

    def test_few(self):
        attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                 'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
        sub_attrs = ['att_1', 'att_4', 'att_3']
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
            returned_attrs = hdf_utils.get_attributes(h5_group, sub_attrs)
            self.assertIsInstance(returned_attrs, dict)
            for key in sub_attrs:
                self.assertTrue(np.all(returned_attrs[key] == attrs[key]))

    def test_all(self):
        attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                 'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
            returned_attrs = hdf_utils.get_attributes(h5_group)
            self.assertIsInstance(returned_attrs, dict)
            for key in attrs.keys():
                self.assertTrue(np.all(returned_attrs[key] == attrs[key]))

    def test_absent_attr(self):
        sub_attrs = ['att_1', 'att_4', 'does_not_exist']
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
            with self.assertRaises(KeyError):
                _ = hdf_utils.get_attributes(h5_group, attr_names=sub_attrs)

    def test_not_hdf_obj(self):
        with self.assertRaises(TypeError):
            _ = hdf_utils.get_attributes(np.arange(4))

    def test_invalid_type_single(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
            with self.assertRaises(TypeError):
                _ = hdf_utils.get_attributes(h5_group, 15)

    def test_invalid_type_multi(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
            with self.assertRaises(TypeError):
                _ = hdf_utils.get_attributes(h5_group, ['att_1', 15])


class TestGetAuxillaryDatasets(TestHDFUtilsBase):

    def test_single(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            h5_pos = h5_f['/Raw_Measurement/Position_Indices']
            [ret_val] = hdf_utils.get_auxiliary_datasets(h5_main, aux_dset_name='Position_Indices')
            self.assertEqual(ret_val, h5_pos)

    def test_single(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            h5_pos = h5_f['/Raw_Measurement/Position_Indices']
            [ret_val] = hdf_utils.get_auxiliary_datasets(h5_main, aux_dset_name='Position_Indices')
            self.assertEqual(ret_val, h5_pos)

    def test_multiple(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            h5_pos_inds = h5_f['/Raw_Measurement/Position_Indices']
            h5_pos_vals = h5_f['/Raw_Measurement/Position_Values']
            ret_val = hdf_utils.get_auxiliary_datasets(h5_main, aux_dset_name=['Position_Indices',
                                                                               'Position_Values'])
            self.assertEqual(set(ret_val), set([h5_pos_inds, h5_pos_vals]))

    def test_all(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = [h5_f['/Raw_Measurement/Position_Indices'],
                        h5_f['/Raw_Measurement/Position_Values'],
                        h5_f['/Raw_Measurement/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/Spectroscopic_Values']]
            ret_val = hdf_utils.get_auxiliary_datasets(h5_main)
            self.assertEqual(set(expected), set(ret_val))

    def test_illegal(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            with self.assertRaises(KeyError):
                _ = hdf_utils.get_auxiliary_datasets(h5_main, aux_dset_name='Does_Not_Exist')

    def test_illegal_dset_type(self):
        with self.assertRaises(TypeError):
            _ = hdf_utils.get_auxiliary_datasets(np.arange(5), aux_dset_name='Does_Not_Exist')

    def test_illegal_target_type(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            with self.assertRaises(TypeError):
                _ = hdf_utils.get_auxiliary_datasets(h5_main, aux_dset_name=14)

    def test_illegal_target_type_list(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            with self.assertRaises(TypeError):
                _ = hdf_utils.get_auxiliary_datasets(h5_main, aux_dset_name=[14, 'Position_Indices'])


class TestGetH5ObjRefs(TestHDFUtilsBase):

    def test_many(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_obj_refs = [h5_f,
                           4.123,
                           np.arange(6),
                           h5_f['/Raw_Measurement/Position_Indices'],
                           h5_f['/Raw_Measurement/source_main-Fitter_000'],
                           h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                           h5_f['/Raw_Measurement/Spectroscopic_Values']]
            chosen_objs = [h5_f['/Raw_Measurement/Position_Indices'],
                           h5_f['/Raw_Measurement/source_main-Fitter_000'],
                           h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices']]

            target_ref_names = ['Position_Indices', 'source_main-Fitter_000', 'Spectroscopic_Indices']

            returned_h5_objs = hdf_utils.get_h5_obj_refs(target_ref_names, h5_obj_refs)

            self.assertEqual(set(chosen_objs), set(returned_h5_objs))

    def test_single(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_obj_refs = [h5_f,
                           4.123,
                           np.arange(6),
                           h5_f['/Raw_Measurement/Position_Indices'],
                           h5_f['/Raw_Measurement/source_main-Fitter_000'],
                           h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                           h5_f['/Raw_Measurement/Spectroscopic_Values']]
            chosen_objs = [h5_f['/Raw_Measurement/Position_Indices']]

            target_ref_names = ['Position_Indices']

            returned_h5_objs = hdf_utils.get_h5_obj_refs(target_ref_names[0], h5_obj_refs)

            self.assertEqual(set(chosen_objs), set(returned_h5_objs))

    def test_non_string_names(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_obj_refs = [h5_f, 4.123, np.arange(6),
                           h5_f['/Raw_Measurement/Position_Indices']]

            target_ref_names = ['Position_Indices', np.arange(6), 4.123]

            with self.assertRaises(TypeError):
                _ = hdf_utils.get_h5_obj_refs(target_ref_names, h5_obj_refs)

    def test_no_hdf5_datasets(self):
        h5_obj_refs = 4.124

        target_ref_names = ['Position_Indices']

        with self.assertRaises(TypeError):
            _ = hdf_utils.get_h5_obj_refs(target_ref_names, h5_obj_refs)

    def test_same_name(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_obj_refs = [h5_f['/Raw_Measurement/source_main-Fitter_001/Spectroscopic_Indices'],
                           h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                           h5_f['/Raw_Measurement/Spectroscopic_Values']]
            expected_objs = [h5_f['/Raw_Measurement/source_main-Fitter_001/Spectroscopic_Indices'],
                             h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices']]

            target_ref_names = ['Spectroscopic_Indices']

            returned_h5_objs = hdf_utils.get_h5_obj_refs(target_ref_names, h5_obj_refs)

            self.assertEqual(set(expected_objs), set(returned_h5_objs))


class TestWriteSimpleAttrs(TestHDFUtilsBase):

    def test_invalid_h5_obj(self):
        with self.assertRaises(TypeError):
            hdf_utils.write_simple_attrs(np.arange(4), {'sds': 3})

    def test_invalid_h5_obj_reg_ref(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            ref_in = hdf_utils.get_attr(h5_main, 'even_rows')
            with self.assertRaises(TypeError):
                hdf_utils.write_simple_attrs(ref_in, {'sds': 3})

    def test_invalid_attrs_dict(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_group = h5_f.create_group('Blah')
            with self.assertRaises(TypeError):
                hdf_utils.write_simple_attrs(h5_group, ['attrs', 1.234, 'should be dict', np.arange(3)])
        os.remove(file_path)

    def test_invalid_val_type_in_dict(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_group = h5_f.create_group('Blah')
            # with self.assertRaises(TypeError):
            #    hdf_utils.write_simple_attrs(h5_group, {'att_1': [{'a': 'b'}]})
        os.remove(file_path)

    def test_key_not_str_strict(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            for attrs in [{15: 'hello'},
                          {None: 23},
                          {15.234: 'blah'},
                          {True: False}]:

                if sys.version_info.major == 3:
                    with self.assertWarns(UserWarning):
                        hdf_utils.write_simple_attrs(h5_f, attrs,
                                                     force_to_str=False)
                else:
                    hdf_utils.write_simple_attrs(h5_f, attrs,
                                                 force_to_str=False)
            self.assertEqual(len(h5_f.attrs.keys()), 0)

        os.remove(file_path)

    def test_key_not_str_relaxed(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)

        with h5py.File(file_path, mode='w') as h5_f:
            for attrs in [{15: 'hello'},
                          {None: 23},
                          {15.234: 'blah'},
                          {True: False}]:

                if sys.version_info.major == 3:
                    with self.assertWarns(UserWarning):
                        hdf_utils.write_simple_attrs(h5_f, attrs,
                                                     force_to_str=True)
                else:
                    hdf_utils.write_simple_attrs(h5_f, attrs,
                                                 force_to_str=True)
                key = list(attrs.keys())[0]
                val = attrs[key]
                key = str(key)
                if not isinstance(val, (str, unicode, Number)):
                    val = str(val)

                self.assertTrue(key in h5_f.attrs.keys())
                self.assertEqual(val, h5_f.attrs[key])

            self.assertEqual(len(h5_f.attrs.keys()), 4)

        os.remove(file_path)

    def test_to_grp(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:

            h5_group = h5_f.create_group('Blah')

            attrs = {'att_1': 'string_val', 'att_2': 1.234, 'att_3': [1, 2, 3.14, 4],
                     'att_4': ['s', 'tr', 'str_3']}

            hdf_utils.write_simple_attrs(h5_group, attrs)

            for key, expected_val in attrs.items():
                self.assertTrue(np.all(hdf_utils.get_attr(h5_group, key) == expected_val))

        os.remove(file_path)

    def test_nested_attrs(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:

            h5_group = h5_f.create_group('Blah')

            attrs = {'att_1': 'string_val',
                     'att_2': {'attr_3': [1, 2, 3.14, 4],
                               'att_4': ['s', 'tr', 'str_3']},
                     'att_5': {'att_6': 4},
                     }
            # with self.assertRaises(ValueError):
            #     hdf_utils.write_simple_attrs(h5_group, attrs)

        os.remove(file_path)

    def test_np_array(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:

            attrs = {'att_1': np.random.rand(4)}

            hdf_utils.write_simple_attrs(h5_f, attrs)

            for key, expected_val in attrs.items():
                self.assertTrue(np.all(hdf_utils.get_attr(h5_f, key) == expected_val))

        os.remove(file_path)

    def test_none_ignored(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:

            attrs = {'att_1': None}

            hdf_utils.write_simple_attrs(h5_f, attrs)

            self.assertTrue('att_1' not in h5_f.attrs.keys())

        os.remove(file_path)

    def test_enum_value(self):
        file_path = 'test.h5'

        class MyEnum(Enum):
            BLUE = 1
            RED = 2
            GREEN = 3

        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:

            val = MyEnum.RED

            key = 'att_1'
            expected = val.name
            attrs = {'att_1': val}

            hdf_utils.write_simple_attrs(h5_f, attrs)

            self.assertEqual(hdf_utils.get_attr(h5_f, key), expected)

        os.remove(file_path)

    def test_space_in_key_removed(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            attrs = {'   before': 1,
                     'after    ': 2,
                     '   before_and_after  ': 3,
                     'inside attr name': 4}

            hdf_utils.write_simple_attrs(h5_f, attrs)

            for key, expected_val in attrs.items():
                self.assertTrue(key.strip() in h5_f.attrs.keys())
                self.assertTrue(hdf_utils.get_attr(h5_f, key) == expected_val)

        os.remove(file_path)

    def test_to_dset(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:

            h5_dset = h5_f.create_dataset('Test', data=np.arange(3))

            attrs = {'att_1': 'string_val',
                     'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4],
                     'att_4': ['str_1', 'str_2', 'str_3']}

            hdf_utils.write_simple_attrs(h5_dset, attrs)

            self.assertEqual(len(h5_dset.attrs), len(attrs))

            for key, expected_val in attrs.items():
                self.assertTrue(np.all(hdf_utils.get_attr(h5_dset, key) == expected_val))

        os.remove(file_path)


class TestIsEditableH5(TestHDFUtilsBase):

    def test_read_only(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement']
            h5_main = h5_f['/Raw_Measurement/Ancillary']
            self.assertFalse(hdf_utils.is_editable_h5(h5_group))
            self.assertFalse(hdf_utils.is_editable_h5(h5_f))
            self.assertFalse(hdf_utils.is_editable_h5(h5_main))

    def test_r_plus(self):
        with h5py.File(data_utils.std_beps_path, mode='r+') as h5_f:
            h5_group = h5_f['/Raw_Measurement']
            h5_main = h5_f['/Raw_Measurement/Ancillary']
            self.assertTrue(hdf_utils.is_editable_h5(h5_group))
            self.assertTrue(hdf_utils.is_editable_h5(h5_f))
            self.assertTrue(hdf_utils.is_editable_h5(h5_main))

    def test_w(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.arange(3))
            h5_group = h5_f.create_group('blah')
            self.assertTrue(hdf_utils.is_editable_h5(h5_group))
            self.assertTrue(hdf_utils.is_editable_h5(h5_f))
            self.assertTrue(hdf_utils.is_editable_h5(h5_dset))

        os.remove(file_path)

    def test_invalid_type(self):
        # wrong kind of object
        with self.assertRaises(TypeError):
            _ = hdf_utils.is_editable_h5(np.arange(4))

    def test_closed_file(self):
        with h5py.File(data_utils.std_beps_path, mode='r+') as h5_f:
            h5_group = h5_f['/Raw_Measurement']

        with self.assertRaises(ValueError):
            _ = hdf_utils.is_editable_h5(h5_group)


class TestLinkH5ObjAsAlias(unittest.TestCase):

    def test_legal(self):
        file_path = 'link_as_alias.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:

            h5_main = h5_f.create_dataset('main', data=np.arange(5))
            h5_anc = h5_f.create_dataset('Ancillary', data=np.arange(3))
            h5_group = h5_f.create_group('Results')

            # Linking to dataset:
            hdf_utils.link_h5_obj_as_alias(h5_main, h5_anc, 'Blah')
            hdf_utils.link_h5_obj_as_alias(h5_main, h5_group, 'Something')
            self.assertEqual(h5_f[h5_main.attrs['Blah']], h5_anc)
            self.assertEqual(h5_f[h5_main.attrs['Something']], h5_group)

            # Linking ot Group:
            hdf_utils.link_h5_obj_as_alias(h5_group, h5_main, 'Center')
            hdf_utils.link_h5_obj_as_alias(h5_group, h5_anc, 'South')
            self.assertEqual(h5_f[h5_group.attrs['Center']], h5_main)
            self.assertEqual(h5_f[h5_group.attrs['South']], h5_anc)

            # Linking to file:
            hdf_utils.link_h5_obj_as_alias(h5_f, h5_main, 'Paris')
            hdf_utils.link_h5_obj_as_alias(h5_f, h5_group, 'France')
            self.assertEqual(h5_f[h5_f.attrs['Paris']], h5_main)
            self.assertEqual(h5_f[h5_f.attrs['France']], h5_group)
        os.remove(file_path)

    def test_not_h5_obj(self):
        file_path = 'link_as_alias.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_group = h5_f.create_group('Results')

            # Non h5 object
            with self.assertRaises(TypeError):
                hdf_utils.link_h5_obj_as_alias(h5_group, np.arange(5), 'Center')

            # H5 reference but not the object
            with self.assertRaises(TypeError):
                hdf_utils.link_h5_obj_as_alias('not_a_dset', h5_group, 'Center')

            with self.assertRaises(TypeError):
                hdf_utils.link_h5_obj_as_alias(h5_group, h5_group, 1.234)

        os.remove(file_path)


class TestLinkH5ObjectAsAttribute(unittest.TestCase):

    def test_legal(self):
        file_path = 'link_h5_objects_as_attrs.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:

            h5_main = h5_f.create_dataset('main', data=np.arange(5))
            h5_anc = h5_f.create_dataset('Ancillary', data=np.arange(3))
            h5_group = h5_f.create_group('Results')

            hdf_utils.link_h5_objects_as_attrs(h5_f, [h5_anc, h5_main, h5_group])
            for exp, name in zip([h5_main, h5_anc, h5_group], ['main', 'Ancillary', 'Results']):
                self.assertEqual(exp, h5_f[h5_f.attrs[name]])

            # Single object
            hdf_utils.link_h5_objects_as_attrs(h5_main, h5_anc)
            self.assertEqual(h5_f[h5_main.attrs['Ancillary']], h5_anc)

            # Linking to a group:
            hdf_utils.link_h5_objects_as_attrs(h5_group, [h5_anc, h5_main])
            for exp, name in zip([h5_main, h5_anc], ['main', 'Ancillary']):
                self.assertEqual(exp, h5_group[h5_group.attrs[name]])

        os.remove(file_path)

    def test_wrong_type(self):
        file_path = 'link_h5_objects_as_attrs.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_main = h5_f.create_dataset('main', data=np.arange(5))

            with self.assertRaises(TypeError):
                hdf_utils.link_h5_objects_as_attrs(h5_main, np.arange(4))

            with self.assertRaises(TypeError):
                hdf_utils.link_h5_objects_as_attrs(np.arange(4), h5_main)

        os.remove(file_path)


class TestValidateH5ObjsInSameFile(unittest.TestCase):

    def test_diff_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path_1 = tmp_dir + 'source.h5'
            file_path_2 = tmp_dir + 'sink.h5'
        data_utils.delete_existing_file(file_path_1)
        h5_f1 = h5py.File(file_path_1, mode='w')
        h5_main = h5_f1.create_dataset('main', data=np.arange(5))
        h5_f2 = h5py.File(file_path_2, mode='w')
        h5_other = h5_f2.create_dataset('other', data=np.arange(5))

        with self.assertRaises(ValueError):
            hdf_utils.validate_h5_objs_in_same_h5_file(h5_main, h5_other)

        #os.remove(file_path_1)
        #os.remove(file_path_2)

    def test_same_file(self):
        file_path = 'test_same_file.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_main = h5_f.create_dataset('main', data=np.arange(5))
            h5_anc = h5_f.create_dataset('Ancillary', data=np.arange(3))
            # Nothing should happen here.
            hdf_utils.validate_h5_objs_in_same_h5_file(h5_main, h5_anc)
        os.remove(file_path)


class TestWriteBookKeepingAttrs(unittest.TestCase):

    def test_file(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            hdf_utils.write_book_keeping_attrs(h5_f)
            data_utils.verify_book_keeping_attrs(self, h5_f)
        os.remove(file_path)

    def test_group(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_g = h5_f.create_group('group')
            hdf_utils.write_book_keeping_attrs(h5_g)
            data_utils.verify_book_keeping_attrs(self, h5_g)
        os.remove(file_path)

    def test_dset(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('dset', data=[1, 2, 3])
            hdf_utils.write_book_keeping_attrs(h5_dset)
            data_utils.verify_book_keeping_attrs(self, h5_dset)
        os.remove(file_path)

    def test_invalid(self):
        with self.assertRaises(TypeError):
            hdf_utils.write_book_keeping_attrs(np.arange(4))


class TestPrintTreeNoMain(unittest.TestCase):

    def test_not_a_group(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)

        with h5py.File(file_path, mode='w') as h5_f:
            dset = h5_f.create_dataset('A_Dataset', data=[1, 2, 3])
            with self.assertRaises(TypeError):
                hdf_utils.print_tree(dset, rel_paths=False)

    def test_single_level_tree(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        expected = ['/']
        with h5py.File(file_path, mode='w') as h5_f:

            obj_name = 'A_Dataset'
            expected.append(0 * '  ' + '├ ' + obj_name)
            _ = h5_f.create_dataset(obj_name, data=[1, 2, 3])

            obj_name = 'B_Group'
            expected.append(0 * '  ' + '├ ' + obj_name)
            expected.append((0 + 1) * '  ' + len(obj_name) * '-')
            _ = h5_f.create_group(obj_name)

            with data_utils.capture_stdout() as get_value:
                hdf_utils.print_tree(h5_f, rel_paths=False)

                actual = get_value()
        expected = '\n'.join(expected) + '\n'
        self.assertEqual(expected, actual)
        os.remove(file_path)

    def test_single_level_rel_paths(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        expected = ['/']
        with h5py.File(file_path, mode='w') as h5_f:

            obj_name = 'A_Dataset'
            expected.append(obj_name)
            _ = h5_f.create_dataset(obj_name, data=[1, 2, 3])

            obj_name = 'B_Group'
            expected.append(obj_name)
            _ = h5_f.create_group(obj_name)

            with data_utils.capture_stdout() as get_value:
                hdf_utils.print_tree(h5_f, rel_paths=True)

                actual = get_value()
        expected = '\n'.join(expected) + '\n'
        self.assertEqual(expected, actual)
        os.remove(file_path)

    def test_multi_level_tree(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        expected = ['/']
        with h5py.File(file_path, mode='w') as h5_f:

            level = 0

            obj_name = 'A_Group'
            expected.append(level * '  ' + '├ ' + obj_name)
            expected.append((level + 1) * '  ' + len(obj_name) * '-')
            grp_1 = h5_f.create_group(obj_name)
            level += 1

            obj_name = 'B_Group'
            expected.append(level * '  ' + '├ ' + obj_name)
            expected.append((level + 1) * '  ' + len(obj_name) * '-')
            grp_2 = grp_1.create_group(obj_name)
            level += 1

            obj_name = 'C_Group'
            expected.append(level * '  ' + '├ ' + obj_name)
            expected.append((level + 1) * '  ' + len(obj_name) * '-')
            grp_3 = grp_2.create_group(obj_name)
            level += 1

            obj_name = 'Y_Dataset'
            expected.append(level * '  ' + '├ ' + obj_name)
            _ = grp_3.create_dataset(obj_name, data=[1, 2, 3])

            obj_name = 'X_Dataset'
            expected.append(0 * '  ' + '├ ' + obj_name)
            _ = h5_f.create_dataset(obj_name, data=[1, 2, 3])

            with data_utils.capture_stdout() as get_value:
                hdf_utils.print_tree(h5_f, rel_paths=False)

                actual = get_value()
        expected = '\n'.join(expected) + '\n'
        self.assertEqual(expected, actual)
        os.remove(file_path)

    def test_multi_level_tree_grp_a(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        expected = []
        with h5py.File(file_path, mode='w') as h5_f:

            obj_name = 'A_Group'
            grp_1 = h5_f.create_group(obj_name)
            # Full path printed for root always
            expected.append(grp_1.name)

            level = 0

            obj_name = 'B_Group'
            expected.append(level * '  ' + '├ ' + obj_name)
            expected.append((level + 1) * '  ' + len(obj_name) * '-')
            grp_2 = grp_1.create_group(obj_name)
            level += 1

            obj_name = 'C_Group'
            expected.append(level * '  ' + '├ ' + obj_name)
            expected.append((level + 1) * '  ' + len(obj_name) * '-')
            grp_3 = grp_2.create_group(obj_name)
            level += 1

            obj_name = 'Y_Dataset'
            expected.append(level * '  ' + '├ ' + obj_name)
            _ = grp_3.create_dataset(obj_name, data=[1, 2, 3])

            obj_name = 'X_Dataset'
            # expected.append(0 * '  ' + '├ ' + obj_name)
            _ = h5_f.create_dataset(obj_name, data=[1, 2, 3])

            with data_utils.capture_stdout() as get_value:
                hdf_utils.print_tree(grp_1, rel_paths=False)

                actual = get_value()
        expected = '\n'.join(expected) + '\n'
        self.assertEqual(expected, actual)
        os.remove(file_path)

    def test_multi_level_tree_grp_b(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        expected = []
        with h5py.File(file_path, mode='w') as h5_f:

            obj_name = 'A_Group'
            grp_1 = h5_f.create_group(obj_name)

            obj_name = 'B_Group'
            grp_2 = grp_1.create_group(obj_name)
            # Full path printed for root always
            expected.append(grp_2.name)

            level = 0

            obj_name = 'C_Group'
            expected.append(level * '  ' + '├ ' + obj_name)
            expected.append((level + 1) * '  ' + len(obj_name) * '-')
            grp_3 = grp_2.create_group(obj_name)
            level += 1

            obj_name = 'Y_Dataset'
            expected.append(level * '  ' + '├ ' + obj_name)
            _ = grp_3.create_dataset(obj_name, data=[1, 2, 3])

            obj_name = 'X_Dataset'
            # expected.append(0 * '  ' + '├ ' + obj_name)
            _ = h5_f.create_dataset(obj_name, data=[1, 2, 3])

            with data_utils.capture_stdout() as get_value:
                hdf_utils.print_tree(grp_2, rel_paths=False)

                actual = get_value()
        expected = '\n'.join(expected) + '\n'
        self.assertEqual(expected, actual)
        os.remove(file_path)

    def test_multi_level_rel_paths_grp_b(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        expected = []
        with h5py.File(file_path, mode='w') as h5_f:

            obj_name = 'A_Group'
            grp_1 = h5_f.create_group(obj_name)

            obj_name = 'B_Group'
            grp_2 = grp_1.create_group(obj_name)
            # Full path printed for root always
            expected.append(grp_2.name)

            obj_name = 'C_Group'
            grp_3 = grp_2.create_group(obj_name)
            expected.append(grp_3.name.replace(grp_2.name + '/', ''))

            obj_name = 'Y_Dataset'
            dset = grp_3.create_dataset(obj_name, data=[1, 2, 3])
            expected.append(dset.name.replace(grp_2.name + '/', ''))

            obj_name = 'X_Dataset'
            _ = h5_f.create_dataset(obj_name, data=[1, 2, 3])

            with data_utils.capture_stdout() as get_value:
                hdf_utils.print_tree(grp_2, rel_paths=True)

                actual = get_value()
        expected = '\n'.join(expected) + '\n'
        self.assertEqual(expected, actual)
        os.remove(file_path)

    def test_multi_level_rel_paths(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        expected = ['/']
        with h5py.File(file_path, mode='w') as h5_f:

            obj_name = 'A_Group'
            grp_1 = h5_f.create_group(obj_name)
            expected.append(grp_1.name[1:])

            obj_name = 'B_Group'
            grp_2 = grp_1.create_group(obj_name)
            expected.append(grp_2.name[1:])

            obj_name = 'C_Group'
            grp_3 = grp_2.create_group(obj_name)
            expected.append(grp_3.name[1:])

            obj_name = 'Y_Dataset'
            dset = grp_3.create_dataset(obj_name, data=[1, 2, 3])
            expected.append(dset.name[1:])

            obj_name = 'X_Dataset'
            dset = h5_f.create_dataset(obj_name, data=[1, 2, 3])
            expected.append(dset.name[1:])

            with data_utils.capture_stdout() as get_value:
                hdf_utils.print_tree(h5_f, rel_paths=True)

                actual = get_value()
        expected = '\n'.join(expected) + '\n'
        self.assertEqual(expected, actual)
        os.remove(file_path)


class TestPrintTreeBEPS(TestHDFUtilsBase):

    def test_root_all_dsets(self):
        level = 0
        expected = ['/',
                    level * '  ' + '├ ' + 'Raw_Measurement',
                    (level + 1) * '  ' + len('Raw_Measurement') * '-']
        level += 1
        expected += [
                    level * '  ' + '├ ' + 'Ancillary',
                    level * '  ' + '├ ' + 'Bias',
                    level * '  ' + '├ ' + 'Cycle',
                    level * '  ' + '├ ' + 'Misc',
                    (level + 1) * '  ' + len('Misc') * '-',
                    level * '  ' + '├ ' + 'Position_Indices',
                    level * '  ' + '├ ' + 'Position_Values',
                    level * '  ' + '├ ' + 'Spectroscopic_Indices',
                    level * '  ' + '├ ' + 'Spectroscopic_Values',
                    level * '  ' + '├ ' + 'X',
                    level * '  ' + '├ ' + 'Y',
                    level * '  ' + '├ ' + 'n_dim_form',
                    level * '  ' + '├ ' + 'source_main']
        level += 1
        for ind in range(2):
            expected += [
                        (level-1) * '  ' + '├ ' + 'source_main-Fitter_00'+str(ind),
                        level * '  ' + len('source_main-Fitter_000') * '-',
                        level * '  ' + '├ ' + 'Spectroscopic_Indices',
                        level * '  ' + '├ ' + 'Spectroscopic_Values',
                        level * '  ' + '├ ' + 'n_dim_form',
                        level * '  ' + '├ ' + 'results_main',
                        ]
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            with data_utils.capture_stdout() as get_value:
                hdf_utils.print_tree(h5_f, rel_paths=False)

                actual = get_value()
        expected = '\n'.join(expected) + '\n'
        self.assertEqual(expected, actual)


class TestCopyAttributes(unittest.TestCase):

    def test_not_h5_dset(self):
        temp_path = 'copy_attributes.h5'
        with h5py.File(temp_path, mode='w') as h5_f:
            h5_grp = h5_f.create_group('Blah')
            with self.assertRaises(TypeError):
                hdf_utils.copy_attributes(h5_grp, np.arange(4))

            with self.assertRaises(TypeError):
                hdf_utils.copy_attributes(np.arange(4), h5_grp)
        os.remove(temp_path)

    def test_file_dset(self):
        file_path = 'test.h5'
        easy_attrs = {'1_string': 'Current', '1_number': 35.23}
        also_easy_attr = {'N_numbers': [1, -53.6, 0.000463]}
        hard_attrs = {'N_strings': np.array(['a', 'bc', 'def'], dtype='S')}
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_f.attrs.update(easy_attrs)
            h5_f.attrs.update(also_easy_attr)
            h5_f.attrs.update(hard_attrs)
            h5_dset = h5_f.create_dataset('Main_01', data=[1, 2, 3])
            hdf_utils.copy_attributes(h5_f, h5_dset)
            for key, val in easy_attrs.items():
                self.assertEqual(val, h5_dset.attrs[key])
            for key, val in also_easy_attr.items():
                self.assertTrue(np.all([x == y for x, y in zip(val, h5_dset.attrs[key])]))
            for key, val in hard_attrs.items():
                self.assertTrue(np.all([x == y for x, y in zip(val, h5_dset.attrs[key])]))
        os.remove(file_path)

    def test_group_dset(self):
        file_path = 'test.h5'
        easy_attrs = {'1_string': 'Current', '1_number': 35.23}
        also_easy_attr = {'N_numbers': [1, -53.6, 0.000463]}
        hard_attrs = {'N_strings': np.array(['a', 'bc', 'def'], dtype='S')}
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_group = h5_f.create_group('Group')
            h5_group.attrs.update(easy_attrs)
            h5_group.attrs.update(also_easy_attr)
            h5_group.attrs.update(hard_attrs)
            h5_dset = h5_f.create_dataset('Main_01', data=[1, 2, 3])
            hdf_utils.copy_attributes(h5_group, h5_dset)
            for key, val in easy_attrs.items():
                self.assertEqual(val, h5_dset.attrs[key])
            for key, val in also_easy_attr.items():
                self.assertTrue(np.all([x == y for x, y in zip(val, h5_dset.attrs[key])]))
            for key, val in hard_attrs.items():
                self.assertTrue(np.all([x == y for x, y in zip(val, h5_dset.attrs[key])]))
        os.remove(file_path)


class TestCopyDataset(TestHDFUtilsBase):

    def validate_copied_dataset(self, h5_f_new, dset_new_name,
                                dset_data, dset_attrs):
        self.assertTrue(dset_new_name in h5_f_new.keys())
        h5_anc_dest = h5_f_new[dset_new_name]
        self.assertIsInstance(h5_anc_dest, h5py.Dataset)
        self.assertTrue(np.allclose(dset_data, h5_anc_dest[()]))
        self.assertEqual(len(dset_attrs),
                         len(h5_anc_dest.attrs.keys()))
        for key, val in dset_attrs.items():
            self.assertEqual(val, h5_anc_dest.attrs[key])

    def base_test(self, exist_dset_same_data=False, use_alias=False,
                  exist_dset_diff_data_shape=False, exist_dset_diff_data=False,
                  exist_grp_inst_dset=False):
        file_path = 'test.h5'
        new_path = 'new.h5'
        data_utils.delete_existing_file(file_path)
        data_utils.delete_existing_file(new_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_source = h5_f.create_dataset('Original', data=[1, 2, 3])
            simple_attrs = {'quantity': 'blah', 'units': 'nA'}
            h5_source.attrs.update(simple_attrs)

            with h5py.File(new_path, mode='w') as h5_f_new:

                if use_alias:
                    alias = 'Duplicate'
                else:
                    alias = 'Original'

                if exist_dset_same_data:
                    _ = h5_f_new.create_dataset(alias, data=[1, 2, 3])
                elif exist_dset_diff_data:
                    _ = h5_f_new.create_dataset(alias, data=[8, 1, 3])
                elif exist_dset_diff_data_shape:
                    _ = h5_f_new.create_dataset(alias,
                                                data=np.random.rand(5, 3))
                elif exist_grp_inst_dset:
                    _ = h5_f_new.create_group(alias)

                if use_alias:
                    al_arg = alias
                else:
                    al_arg = None

                func = hdf_utils.copy_dataset
                args = [h5_source, h5_f_new]
                kwargs = {'alias': al_arg, 'verbose': False}

                if exist_dset_diff_data or exist_dset_diff_data_shape:
                    with self.assertRaises(ValueError):
                        _ = func(*args, **kwargs)
                elif exist_grp_inst_dset:
                    with self.assertRaises(TypeError):
                        _ = func(*args, **kwargs)
                else:
                    _ = func(*args, **kwargs)

                if not exist_dset_diff_data_shape and not exist_dset_diff_data\
                        and not exist_grp_inst_dset:
                    self.assertEqual(len(h5_f_new.keys()), 1)
                    self.validate_copied_dataset(h5_f_new, alias,
                                                 h5_source[()], simple_attrs)

        os.remove(file_path)
        os.remove(new_path)

    def test_exact_copy(self):
        self.base_test(exist_dset_same_data=False, use_alias=False,
                       exist_dset_diff_data_shape=False,
                       exist_dset_diff_data=False, exist_grp_inst_dset=False)

    def test_copy_w_alias(self):
        self.base_test(exist_dset_same_data=False, use_alias=True,
                       exist_dset_diff_data_shape=False,
                       exist_dset_diff_data=False, exist_grp_inst_dset=False)

    def test_existing_group_same_name(self):
        self.base_test(exist_dset_same_data=False, use_alias=False,
                       exist_dset_diff_data_shape=False,
                       exist_dset_diff_data=False, exist_grp_inst_dset=True)

    def test_existing_dset_same_name_data(self):
        self.base_test(exist_dset_same_data=True, use_alias=False,
                       exist_dset_diff_data_shape=False,
                       exist_dset_diff_data=False, exist_grp_inst_dset=False)

    def test_existing_dset_same_name_diff_data_shape(self):
        self.base_test(exist_dset_same_data=False, use_alias=False,
                       exist_dset_diff_data_shape=True,
                       exist_dset_diff_data=False, exist_grp_inst_dset=False)

    def test_existing_dset_same_name_diff_data(self):
        self.base_test(exist_dset_same_data=False, use_alias=False,
                       exist_dset_diff_data_shape=False,
                       exist_dset_diff_data=True, exist_grp_inst_dset=False)


"""
    def test_linking_main_plus_other_dsets(self):
        file_path = 'check_and_link_ancillary.h5'
        data_utils.delete_existing_file(file_path)
        shutil.copy(data_utils.std_beps_path, file_path)
        with h5py.File(file_path, mode='r+') as h5_f:
            h5_grp = h5_f['Raw_Measurement']
            h5_dset_source = h5_grp['Ancillary']
            h5_main = h5_grp['source_main']
            att_names = ['Spectroscopic_Values', 'Position_Indices', 'X', 'Y']
            expected = [h5_grp[name] for name in att_names]
            hdf_utils.check_and_link_ancillary(h5_dset_source, att_names,
                                               h5_main=h5_main,
                                               anc_refs=expected[2:])
            for key, val in h5_dset_source.attrs.items():
                print(key, val)
            self.assertEqual(len(h5_dset_source.attrs.keys()), len(att_names))
            self.assertEqual(set(att_names), set(h5_dset_source.attrs.keys()))
            for name, exp_val in zip(att_names, expected):
                actual = h5_dset_source.attrs[name]
                self.assertIsInstance(actual, h5py.Reference)
                self.assertEqual(h5_f[actual], exp_val)
        os.remove(file_path)
"""


class TestCopyLinkedObjects(TestHDFUtilsBase):

    def validate_copied_dataset(self, h5_f_new, h5_dest, dset_new_name,
                                dset_data, dset_attrs):
        self.assertTrue(dset_new_name in h5_f_new.keys())
        h5_anc_dest = h5_f_new[dset_new_name]
        self.assertIsInstance(h5_anc_dest, h5py.Dataset)
        self.assertTrue(np.allclose(dset_data, h5_anc_dest[()]))
        self.assertEqual(len(dset_attrs),
                         len(h5_anc_dest.attrs.keys()))
        for key, val in dset_attrs.items():
            self.assertEqual(val, h5_anc_dest.attrs[key])
        self.assertTrue(dset_new_name in h5_dest.attrs.keys())
        self.assertEqual(h5_f_new[h5_dest.attrs[dset_new_name]],
                         h5_anc_dest)

    def base_two_dsets_simple_attrs(self, exist_dset_same_data=False,
                                    exist_dset_diff_data_shape=False,
                                    exist_dset_diff_data=False,
                                    exist_grp_inst_dset=False):
        file_path = 'test.h5'
        new_path = 'new.h5'
        data_utils.delete_existing_file(file_path)
        data_utils.delete_existing_file(new_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_source = h5_f.create_dataset('Main', data=[1, 2, 3])
            simple_attrs = {'quantity': 'blah', 'units': 'nA'}
            h5_source.attrs.update(simple_attrs)

            h5_anc_1 = h5_f.create_dataset('Anc_1', data=[4, 5, 6])
            anc_1_attrs = {'a': 1, 'b': 3}
            h5_anc_1.attrs.update(anc_1_attrs)

            h5_anc_2 = h5_f.create_dataset('Anc_2', data=[7, 8, 9])
            anc_2_attrs = {'p': 78, 'j': 8}
            h5_anc_2.attrs.update(anc_2_attrs)

            h5_source.attrs['Pos_Inds'] = h5_anc_1.ref
            h5_source.attrs['Pos_Vals'] = h5_anc_2.ref

            with h5py.File(new_path, mode='w') as h5_f_new:
                h5_dest = h5_f_new.create_dataset('Duplicate', data=[1, 2, 3])

                if exist_dset_same_data:
                    _ = h5_f_new.create_dataset('Pos_Vals', data=[7, 8, 9])
                elif exist_dset_diff_data:
                    _ = h5_f_new.create_dataset('Pos_Vals', data=[8, 1, 3])
                elif exist_dset_diff_data_shape:
                    _ = h5_f_new.create_dataset('Pos_Vals',
                                                data=np.random.rand(5, 3))
                elif exist_grp_inst_dset:
                    _ = h5_f_new.create_group('Pos_Vals')

                if sys.version_info.major == 3 and exist_dset_same_data:
                    with self.assertWarns(UserWarning):
                        hdf_utils.copy_linked_objects(h5_source, h5_dest,
                                                      verbose=False)
                elif exist_dset_diff_data or exist_dset_diff_data_shape:
                    with self.assertRaises(ValueError):
                        hdf_utils.copy_linked_objects(h5_source, h5_dest,
                                                      verbose=False)
                elif exist_grp_inst_dset:
                    with self.assertRaises(TypeError):
                        hdf_utils.copy_linked_objects(h5_source, h5_dest,
                                                      verbose=False)
                else:
                    hdf_utils.copy_linked_objects(h5_source, h5_dest,
                                                  verbose=False)

                if not exist_dset_diff_data_shape and not exist_dset_diff_data \
                        and not exist_grp_inst_dset:
                    self.assertEqual(len(h5_f_new.keys()), 3)

                    self.validate_copied_dataset(h5_f_new, h5_dest, 'Pos_Inds',
                                                 h5_anc_1[()], anc_1_attrs)

                    self.validate_copied_dataset(h5_f_new, h5_dest, 'Pos_Vals',
                                                 h5_anc_2[()], anc_2_attrs)

        os.remove(file_path)
        os.remove(new_path)

    def test_two_dsets_simple_attrs_empty_dest(self):
        self.base_two_dsets_simple_attrs(exist_dset_same_data=False,
                                         exist_dset_diff_data_shape=False,
                                         exist_dset_diff_data=False,
                                         exist_grp_inst_dset=False)

    def test_existing_anc_dset_same_data_no_attrs(self):
        self.base_two_dsets_simple_attrs(exist_dset_same_data=True,
                                         exist_dset_diff_data_shape=False,
                                         exist_dset_diff_data=False,
                                         exist_grp_inst_dset=False)

    def test_existing_anc_dset_diff_data(self):
        self.base_two_dsets_simple_attrs(exist_dset_same_data=False,
                                         exist_dset_diff_data_shape=False,
                                         exist_dset_diff_data=True,
                                         exist_grp_inst_dset=False)

    def test_existing_anc_dset_diff_data_shape(self):
        self.base_two_dsets_simple_attrs(exist_dset_same_data=False,
                                         exist_dset_diff_data_shape=True,
                                         exist_dset_diff_data=False,
                                         exist_grp_inst_dset=False)

    def test_existing_group_instead_of_det(self):
        self.base_two_dsets_simple_attrs(exist_dset_same_data=False,
                                         exist_dset_diff_data_shape=False,
                                         exist_dset_diff_data=False,
                                         exist_grp_inst_dset=True)

    def test_target_group(self):
        file_path = 'test.h5'
        new_path = 'new.h5'
        data_utils.delete_existing_file(file_path)
        data_utils.delete_existing_file(new_path)

        with h5py.File(file_path, mode='w') as h5_f:
            h5_source = h5_f.create_dataset('Main', data=[1, 2, 3])
            simple_attrs = {'quantity': 'blah', 'units': 'nA'}
            h5_source.attrs.update(simple_attrs)

            h5_anc_1 = h5_f.create_dataset('Anc_1', data=[4, 5, 6])
            anc_1_attrs = {'a': 1, 'b': 3}
            h5_anc_1.attrs.update(anc_1_attrs)

            h5_anc_2 = h5_f.create_dataset('Anc_2', data=[7, 8, 9])
            anc_2_attrs = {'p': 78, 'j': 8}
            h5_anc_2.attrs.update(anc_2_attrs)

            h5_source.attrs['Pos_Inds'] = h5_anc_1.ref
            h5_source.attrs['Pos_Vals'] = h5_anc_2.ref

            with h5py.File(new_path, mode='w') as h5_f_new:
                h5_dest_gp = h5_f_new.create_group('Target')

                hdf_utils.copy_linked_objects(h5_source, h5_dest_gp,
                                              verbose=False)

                self.assertTrue('Pos_Inds' in h5_dest_gp)
                self.assertTrue('Pos_Vals' in h5_dest_gp)

                self.assertFalse('Pos_Inds' in h5_dest_gp.parent)
                self.assertFalse('Pos_Vals' in h5_dest_gp.parent)

        os.remove(file_path)
        os.remove(new_path)


class TestFindDataset(TestHDFUtilsBase):

    def test_legal(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/']
            with self.assertRaises(TypeError):
                ret_val = hdf_utils.find_dataset(h5_group, np.arange(4))

    def test_invalid_type_dset(self):
        with self.assertRaises(TypeError):
            _ = hdf_utils.find_dataset(4.324, 'Spectroscopic_Indices')

    def test_illegal(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/']
            ret_val = hdf_utils.find_dataset(h5_group, 'Does_Not_Exist')
            self.assertEqual(len(ret_val), 0)


class TestWriteDictToH5Group(unittest.TestCase):

    def test_not_h5_group_object(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'write_dict_to_h5_group.h5'
            with h5py.File(file_path, mode='w') as h5_file:
                h5_dset = h5_file.create_dataset("dataset", data=[1, 2, 3])
                with self.assertRaises(TypeError):
                    _ = hdf_utils.write_dict_to_h5_group(h5_dset, {'a': 1}, 'blah')

    def test_metadata_not_dict(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'write_dict_to_h5_group.h5'
            with h5py.File(file_path, mode='w') as h5_file:
                with self.assertRaises(TypeError):
                    _ = hdf_utils.write_dict_to_h5_group(h5_file,
                                                         ['not', 'dict'],
                                                         'blah')

    def test_not_valid_group_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'write_dict_to_h5_group.h5'
            with h5py.File(file_path, mode='w') as h5_file:
                with self.assertRaises(ValueError):
                    _ = hdf_utils.write_dict_to_h5_group(h5_file,
                                                         {'s': 1},
                                                         '   ')
                with self.assertRaises(TypeError):
                    _ = hdf_utils.write_dict_to_h5_group(h5_file,
                                                         {'s': 1},
                                                         [1, 4])

    def test_group_name_already_exists(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'write_dict_to_h5_group.h5'
            with h5py.File(file_path, mode='w') as h5_file:
                _ = h5_file.create_dataset("dataset", data=[1, 2, 3])
                with self.assertRaises(ValueError):
                    _ = hdf_utils.write_dict_to_h5_group(h5_file,
                                                         {'a': 1},
                                                         'dataset')

    def test_metadata_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'write_dict_to_h5_group.h5'
            with h5py.File(file_path, mode='w') as h5_file:
                ret_val = hdf_utils.write_dict_to_h5_group(h5_file, {}, 'blah')
                self.assertEqual(ret_val, None)
                self.assertTrue(len(h5_file.keys()) == 0)

    def test_metadata_is_nested(self):
        metadata = {'a': 4, 'b': {'c': 2.353, 'd': 'nested'}}
        flat_md = metadata
        group_name = 'blah'
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'write_dict_to_h5_group.h5'
            with h5py.File(file_path, mode='w') as h5_file:
                h5_grp = hdf_utils.write_dict_to_h5_group(h5_file,
                                                          metadata, group_name)
                self.assertIsInstance(h5_grp, h5py.Group)
                grp_name = h5_grp.name.split('/')[-1]
                # self.assertEqual(grp_name, group_name)
                # self.assertEqual(len(h5_grp.attrs.keys()), len(flat_md))
                # for key, val in flat_md.items():
                #    self.assertEqual(val, hdf_utils.get_attr(h5_grp, key))

    def test_metadata_is_flat(self):
        metadata = {'a': 4, 'b': 'hello'}
        group_name = 'blah'
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'write_dict_to_h5_group.h5'
            with h5py.File(file_path, mode='w') as h5_file:
                h5_grp = hdf_utils.write_dict_to_h5_group(h5_file,
                                                          metadata, group_name)
                self.assertIsInstance(h5_grp, h5py.Group)
                grp_name = h5_grp.name.split('/')[-1]
                self.assertEqual(grp_name, group_name)
                self.assertEqual(len(h5_grp.attrs.keys()), len(metadata))
                for key, val in metadata.items():
                    self.assertEqual(val, hdf_utils.get_attr(h5_grp, key))


if __name__ == '__main__':
    unittest.main()
