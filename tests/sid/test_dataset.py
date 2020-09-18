# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 17:07:16 2020

@author: Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, \
    absolute_import
import unittest
import sys
import numpy as np
import dask.array as da

sys.path.append("../../sidpy/")
from sidpy.sid.dimension import Dimension, DimensionTypes
from sidpy.sid.dataset import DataTypes, Dataset

if sys.version_info.major == 3:
    unicode = str


class TestDatasetFromArray(unittest.TestCase):

    def test_std_inputs(self):
        # verify generic properties, dimensions, etc.
        # Consider putting such validation in a helper function
        # so that other tests can use as well
        pass


class TestDatasetConstructor(unittest.TestCase):

    def test_minimal_inputs(self):
        pass

    def test_all_inputs(self):
        pass

    def test_user_defined_parms(self):
        pass

    def test_invalid_main_types(self):
        pass

    def test_numpy_array_input(self):
        pass

    def test_dask_array_input(self):
        pass

    def test_list_input(self):
        pass

    def test_1d_main_data(self):
        pass

    def test_2d_main_data(self):
        pass

    def test_3d_main_data(self):
        pass

    def test_4d_main_data(self):
        pass

    def test_dimensions_not_matching_main(self):
        pass

    def test_unknown_data_type(self):
        pass

    def test_enum_data_type(self):
        pass

    def test_string_data_type(self):
        pass


class TestDatasetRepr(unittest.TestCase):

    def test_minimal_inputs(self):
        # Just the main dataset
        pass

    def test_fully_configured(self):
        # Main dataset + customized arrays
        # should at least say that metadata exists
        pass

    def test_user_defined_parameters(self):
        # self.blah = 14. Will / should this get printed
        pass


class TestLikeData(unittest.TestCase):

    def test_minimal_inputs(self):
        # Just the main dataset
        pass

    def test_all_customized_properties(self):
        pass


class TestCopy(unittest.TestCase):

    def test_minimal_inputs(self):
        # Just the main dataset
        pass

    def test_all_customized_properties(self):
        pass


class TestRenameDimension(unittest.TestCase):

    def test_valid_index_and_name(self):
        pass

    def test_invalid_index_object_type(self):
        pass

    def test_index_out_of_bounds(self):
        pass

    def test_invalid_name_object_types(self):
        pass

    def test_empty_name_string(self):
        pass

    def test_existing_name(self):
        pass


class TestSetDimension(unittest.TestCase):

    def test_valid_index_and_dim_obj(self):
        pass

    def test_invalid_dim_object(self):
        pass

    # validity of index tested in TestRenameDimension


class TestViewMetadata(unittest.TestCase):

    def test_default_empty_metadata(self):
        pass

    def test_entered_metadata(self):
        pass


class TestViewOriginalMetadata(unittest.TestCase):

    def test_default_empty_metadata(self):
        pass

    def test_entered_metadata(self):
        pass

class TestViewOriginalMetadata(unittest.TestCase):

    def test_valid(self):
        pass

if __name__ == '__main__':
    unittest.main()
