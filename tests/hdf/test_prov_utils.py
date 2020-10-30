# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import os
import sys
import h5py
import numpy as np

sys.path.append("../../sidpy/")
from sidpy.hdf import prov_utils

from .test_hdf_utils import TestHDFUtilsBase
from . import data_utils


if sys.version_info.major == 3:
    unicode = str

class TestAssignGroupIndex(TestHDFUtilsBase):

    def test_existing(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement']
            ret_val = prov_utils.assign_group_index(h5_group, 'source_main-Fitter')
            self.assertEqual(ret_val, 'source_main-Fitter_002')

    def test_new(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement']
            ret_val = prov_utils.assign_group_index(h5_group, 'blah_')
            self.assertEqual(ret_val, 'blah_000')

    def test_invalid_dtypes(self):
        with self.assertRaises(TypeError):
            _ = prov_utils.assign_group_index("not a dataset", 'blah_')

        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement']
            with self.assertRaises(TypeError):
                _ = prov_utils.assign_group_index(h5_group, 1.24)


class TestCreateIndexedGroup(unittest.TestCase):

    def test_first_group(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_group = prov_utils.create_indexed_group(h5_f, 'Hello')
            self.assertIsInstance(h5_group, h5py.Group)
            self.assertEqual(h5_group.name, '/Hello_000')
            self.assertEqual(h5_group.parent, h5_f)
            data_utils.verify_book_keeping_attrs(self, h5_group)

            h5_sub_group = prov_utils.create_indexed_group(h5_group, 'Test')
            self.assertIsInstance(h5_sub_group, h5py.Group)
            self.assertEqual(h5_sub_group.name, '/Hello_000/Test_000')
            self.assertEqual(h5_sub_group.parent, h5_group)
            data_utils.verify_book_keeping_attrs(self, h5_sub_group)
        os.remove(file_path)

    def test_second(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_group_1 = prov_utils.create_indexed_group(h5_f, 'Hello')
            self.assertIsInstance(h5_group_1, h5py.Group)
            self.assertEqual(h5_group_1.name, '/Hello_000')
            self.assertEqual(h5_group_1.parent, h5_f)
            data_utils.verify_book_keeping_attrs(self, h5_group_1)

            h5_group_2 = prov_utils.create_indexed_group(h5_f, 'Hello')
            self.assertIsInstance(h5_group_2, h5py.Group)
            self.assertEqual(h5_group_2.name, '/Hello_001')
            self.assertEqual(h5_group_2.parent, h5_f)
            data_utils.verify_book_keeping_attrs(self, h5_group_2)
        os.remove(file_path)

    def test_w_suffix_(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_group = prov_utils.create_indexed_group(h5_f, 'Hello_')
            self.assertIsInstance(h5_group, h5py.Group)
            self.assertEqual(h5_group.name, '/Hello_000')
            self.assertEqual(h5_group.parent, h5_f)
            data_utils.verify_book_keeping_attrs(self, h5_group)
        os.remove(file_path)

    def test_empty_base_name(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            with self.assertRaises(ValueError):
                _ = prov_utils.create_indexed_group(h5_f, '    ')
        os.remove(file_path)

    def test_create_indexed_group_invalid_types(self):
        with self.assertRaises(TypeError):
            _ = prov_utils.create_indexed_group(np.arange(4), "fddfd")

        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            with self.assertRaises(TypeError):
                _ = prov_utils.create_indexed_group(h5_f, 1.2343)
        os.remove(file_path)


class TestCreateResultsGroup(unittest.TestCase):

    def test_first(self):
        self.helper_first()

    def test_dash_in_name(self):
        self.helper_first(add_dash_to_name=True)

    def helper_first(self, add_dash_to_name=False):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=[1, 2, 3])
            if add_dash_to_name:
                h5_group = prov_utils.create_results_group(h5_dset, 'Some-Tool')
                tool_name = 'Some_Tool'
            else:
                tool_name = 'Tool'
                h5_group = prov_utils.create_results_group(h5_dset, tool_name)
            self.assertIsInstance(h5_group, h5py.Group)
            self.assertEqual(h5_group.name, '/Main-' + tool_name + '_000')
            self.assertEqual(h5_group.parent, h5_f)
            data_utils.verify_book_keeping_attrs(self, h5_group)

            h5_dset = h5_group.create_dataset('Main_Dataset', data=[1, 2, 3])
            h5_sub_group = prov_utils.create_results_group(h5_dset, 'SHO_Fit')
            self.assertIsInstance(h5_sub_group, h5py.Group)
            self.assertEqual(h5_sub_group.name, '/Main-' + tool_name + '_000/Main_Dataset-SHO_Fit_000')
            self.assertEqual(h5_sub_group.parent, h5_group)
            data_utils.verify_book_keeping_attrs(self, h5_sub_group)
        os.remove(file_path)

    def test_second(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=[1, 2, 3])
            h5_group = prov_utils.create_results_group(h5_dset, 'Tool')
            self.assertIsInstance(h5_group, h5py.Group)
            self.assertEqual(h5_group.name, '/Main-Tool_000')
            self.assertEqual(h5_group.parent, h5_f)
            data_utils.verify_book_keeping_attrs(self, h5_group)

            h5_sub_group = prov_utils.create_results_group(h5_dset, 'Tool')
            self.assertIsInstance(h5_sub_group, h5py.Group)
            self.assertEqual(h5_sub_group.name, '/Main-Tool_001')
            self.assertEqual(h5_sub_group.parent, h5_f)
            data_utils.verify_book_keeping_attrs(self, h5_sub_group)
        os.remove(file_path)

    def test_empty_tool_name(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=[1, 2, 3])
            with self.assertRaises(ValueError):
                _ = prov_utils.create_results_group(h5_dset, '   ')
        os.remove(file_path)

    def test_invalid_types(self):
        with self.assertRaises(TypeError):
            _ = prov_utils.create_results_group("not a dataset", 'Tool')

        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            with self.assertRaises(TypeError):
                _ = prov_utils.create_results_group(h5_f, 'Tool')

            h5_dset = h5_f.create_dataset('Main', data=[1, 2, 3])
            with self.assertRaises(TypeError):
                _ = prov_utils.create_results_group(h5_dset, 'Tool',
                                                   h5_parent_group='not_group')

        os.remove(file_path)
        
    def test_different_file(self):
        file_path = 'test.h5'
        new_path = 'new.h5'
        data_utils.delete_existing_file(file_path)
        data_utils.delete_existing_file(new_path)

        with h5py.File(file_path, mode='w') as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=[1, 2, 3])
            # Ensuring that index is calculated at destination, not source:
            _ = h5_f.create_group('Main-Tool_000')

            with h5py.File(new_path, mode='w') as h5_f_new:
                _ = h5_f_new.create_group('Main-Tool_000')

                h5_group = prov_utils.create_results_group(h5_dset, 'Tool',
                                                          h5_parent_group=h5_f_new)

                self.assertIsInstance(h5_group, h5py.Group)
                self.assertEqual(h5_group.name, '/Main-Tool_001')
                self.assertEqual(h5_group.parent, h5_f_new)
                self.assertNotEqual(h5_dset.file, h5_group.file)
                data_utils.verify_book_keeping_attrs(self, h5_group)

        os.remove(file_path)
        os.remove(new_path)


class TestFindResultsGroup(TestHDFUtilsBase):

    def test_legal(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected_groups = [
                h5_f['/Raw_Measurement/source_main-Fitter_000'],
                h5_f['/Raw_Measurement/source_main-Fitter_001']]
            ret_val = prov_utils.find_results_groups(h5_main, 'Fitter')
            self.assertEqual(set(ret_val), set(expected_groups))

    def test_no_dset(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            with self.assertRaises(TypeError):
                _ = prov_utils.find_results_groups(h5_f, 'Fitter')

    def test_not_string(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            with self.assertRaises(TypeError):
                _ = prov_utils.find_results_groups(h5_main, np.arange(5))

    def test_no_such_tool(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            ret_val = prov_utils.find_results_groups(h5_main, 'Blah')
            self.assertEqual(len(ret_val), 0)

    def test_results_in_diff_file(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)

        new_path = 'new.h5'
        data_utils.delete_existing_file(new_path)

        with h5py.File(file_path, mode='w') as h5_f:
            h5_main = h5_f.create_dataset('Main', data=[1, 2, 3])
            with h5py.File(new_path, mode='w') as h5_f_2:
                grp_1 = h5_f_2.create_group('Main-Tool_000')
                grp_2 = h5_f_2.create_group('Main-Tool_001')
                grps = prov_utils.find_results_groups(h5_main, 'Tool',
                                                     h5_parent_group=h5_f_2)
                self.assertEqual(set([grp_1, grp_2]), set(grps))

        os.remove(file_path)
        os.remove(new_path)

    def test_results_in_diff_file_invalid_type(self):
        file_path = 'test.h5'
        data_utils.delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_main = h5_f.create_dataset('Main', data=[1, 2, 3])
            with self.assertRaises(TypeError):
                _ = prov_utils.find_results_groups(h5_main, 'Tool',
                                                  h5_parent_group=h5_main)

        os.remove(file_path)
        
    
class TestCheckForMatchingAttrs(TestHDFUtilsBase):

    def test_dset_no_attrs(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            self.assertTrue(prov_utils.check_for_matching_attrs(h5_main, new_parms=None))

    def test_dset_matching_attrs(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'units': 'A', 'quantity':'Current'}
            self.assertTrue(prov_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_dset_one_mismatched_attrs(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'units': 'A', 'blah': 'meh'}
            self.assertFalse(prov_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_grp(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000']
            attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
            self.assertTrue(prov_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_grp_mismatched_types_01(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000']
            attrs = {'att_4': 'string_val'}
            self.assertFalse(prov_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_grp_mismatched_types_02(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000']
            attrs = {'att_1': ['str_1', 'str_2', 'str_3']}
            self.assertFalse(prov_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_grp_mismatched_types_03(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000']
            attrs = {'att_4': [1, 4.234, 'str_3']}
            self.assertFalse(prov_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_grp_mismatched_types_04(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000']
            attrs = {'att_4': [1, 4.234, 45]}
            self.assertFalse(prov_utils.check_for_matching_attrs(h5_main, new_parms=attrs))


class TestCheckForOld(TestHDFUtilsBase):

    def test_invalid_types(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            with self.assertRaises(TypeError):
                _ = prov_utils.check_for_old("h5_main", "blah")

            with self.assertRaises(TypeError):
                _ = prov_utils.check_for_old(np.arange(4), "blah")

            with self.assertRaises(TypeError):
                _ = prov_utils.check_for_old(h5_main, 1.234)

            with self.assertRaises(TypeError):
                _ = prov_utils.check_for_old(h5_main, 'Fitter',
                                            new_parms="not_a_dictionary")

            with self.assertRaises(TypeError):
                _ = prov_utils.check_for_old(h5_main, 'Fitter',
                                            target_dset=1.234)

    def test_valid_target_dset(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'units': ['V'], 'labels': ['Bias']}
            dset_name = 'Spectroscopic_Indices'
            groups = prov_utils.check_for_old(h5_main, 'Fitter',
                                             new_parms=attrs,
                                             target_dset=dset_name,
                                             verbose=False)
            groups = set(groups)
            self.assertEqual(groups, set([h5_f['/Raw_Measurement/source_main-Fitter_000/'],
                                          h5_f['/Raw_Measurement/source_main-Fitter_001/']]))

    def test_invalid_target_dset(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2',
                                                      'str_3']}
            ret = prov_utils.check_for_old(h5_main, 'Fitter', new_parms=attrs,
                                          target_dset='Does_not_exist')
            self.assertEqual(ret, [])

    def test_exact_match(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
            [h5_ret_grp] = prov_utils.check_for_old(h5_main, 'Fitter',
                                                   new_parms=attrs,
                                                   target_dset=None)
            self.assertEqual(h5_ret_grp, h5_f['/Raw_Measurement/source_main-Fitter_000'])

    def test_subset_but_match(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
            [h5_ret_grp] = prov_utils.check_for_old(h5_main, 'Fitter',
                                                   new_parms=attrs,
                                                   target_dset=None)
            self.assertEqual(h5_ret_grp, h5_f['/Raw_Measurement/source_main-Fitter_000'])

    def test_exact_match_02(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_1': 'other_string_val', 'att_2': 5.4321,
                     'att_3': [4, 1, 3], 'att_4': ['s', 'str_2', 'str_3']}
            [h5_ret_grp] = prov_utils.check_for_old(h5_main, 'Fitter',
                                                   new_parms=attrs,
                                                   target_dset=None)
            self.assertEqual(h5_ret_grp, h5_f['/Raw_Measurement/source_main-Fitter_001'])

    def test_fail_01(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_1': [4, 1, 3], 'att_2': ['s', 'str_2', 'str_3'],
                     'att_3': 'other_string_val', 'att_4': 5.4321}
            ret_val = prov_utils.check_for_old(h5_main, 'Fitter',
                                              new_parms=attrs, target_dset=None)
            self.assertIsInstance(ret_val, list)
            self.assertEqual(len(ret_val), 0)

    def test_fail_02(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_x': [4, 1, 3], 'att_z': ['s', 'str_2', 'str_3'],
                     'att_y': 'other_string_val', 'att_4': 5.4321}
            ret_val = prov_utils.check_for_old(h5_main, 'Fitter',
                                              new_parms=attrs, target_dset=None)
            self.assertIsInstance(ret_val, list)
            self.assertEqual(len(ret_val), 0)


class TestGetSourceDataset(TestHDFUtilsBase):

    def test_legal(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            h5_groups = [h5_f['/Raw_Measurement/source_main-Fitter_000'],
                        h5_f['/Raw_Measurement/source_main-Fitter_001']]
            h5_main = h5_f['/Raw_Measurement/source_main']
            for h5_grp in h5_groups:
                self.assertEqual(h5_main, prov_utils.get_source_dataset(h5_grp))

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            _ = prov_utils.get_source_dataset('/Raw_Measurement/Misc')

    def test_illegal(self):
        with h5py.File(data_utils.std_beps_path, mode='r') as h5_f:
            with self.assertRaises(ValueError):
                _ = prov_utils.get_source_dataset(h5_f['/Raw_Measurement/Misc'])


if __name__ == '__main__':
    unittest.main()