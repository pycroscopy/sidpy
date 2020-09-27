# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 17:07:16 2020

@author: Suhas Somnath, Gerd Duscher
"""
from __future__ import division, print_function, unicode_literals, \
    absolute_import
import unittest
import sys
import numpy as np
import dask.array as da
import string

sys.path.append("../../sidpy/")
from sidpy.sid.dimension import Dimension, DimensionTypes
from sidpy.sid.dataset import DataTypes, Dataset

if sys.version_info.major == 3:
    unicode = str


generic_attributes = ['title', 'quantity', 'units', 'modality', 'source']


class TestDatasetFromArray(unittest.TestCase):

    def test_std_inputs(self):
        # verify generic properties, dimensions, etc.
        values = np.random.random([4, 5, 6])
        descriptor = Dataset.from_array(values)

        for expected, actual in zip(values, descriptor):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))

        self.assertTrue(np.all([hasattr(descriptor, att) for att in generic_attributes]))

        self.assertTrue(np.all([getattr(descriptor, att) == 'generic' for att in generic_attributes]))

        self.assertEqual(descriptor.data_type, DataTypes.UNKNOWN)

        self.assertEqual(descriptor.metadata, {})
        self.assertEqual(descriptor.original_metadata, {})

        for dim in range(len(values.shape)):
            self.assertEqual(getattr(descriptor, string.ascii_lowercase[dim]),
                             getattr(descriptor, 'dim_{}'.format(dim)))

        self.assertFalse(hasattr(descriptor, 'dim_{}'.format(len(values.shape))))
        self.assertFalse(hasattr(descriptor, string.ascii_lowercase[len(values.shape)]))


class TestDatasetConstructor(unittest.TestCase):

    def test_minimal_inputs(self):
        """ test minimum input requirement of an array like object
        """
        with self.assertRaises(TypeError):
            Dataset.from_array()
        descriptor = Dataset.from_array(np.arange(3))
        self.assertIsInstance(descriptor, Dataset)

    def test_all_inputs(self):

        descriptor = Dataset.from_array(np.arange(3), name='test')
        self.assertEqual(descriptor.title, 'test')

    def test_user_defined_parms(self):
        descriptor = Dataset.from_array(np.arange(3), name='test')

        for att in generic_attributes:
            setattr(descriptor, att, 'test')
        self.assertTrue(np.all([getattr(descriptor, att) == 'test' for att in generic_attributes]))

        test_dict = {0: 'test'}
        descriptor.metadata = test_dict.copy()
        descriptor.original_metadata = test_dict.copy()
        self.assertEqual(descriptor.metadata, test_dict)
        self.assertEqual(descriptor.original_metadata, test_dict)

    def test_invalid_main_types(self):
        """
        anything that is not recognized by dask will make an empty dask array
        but name has to be a string
        """
        descriptor = Dataset.from_array(DataTypes.UNKNOWN)
        self.assertEqual(descriptor.shape, ())

        descriptor = Dataset.from_array('test')
        self.assertEqual(descriptor.shape, ())

        descriptor = Dataset.from_array(1)
        self.assertEqual(descriptor.shape, ())

        with self.assertRaises(ValueError):
            Dataset.from_array(1, 1)

    def test_numpy_array_input(self):
        x = np.ones([3, 4, 5])
        descriptor = Dataset.from_array(x, name='test')
        self.assertEqual(descriptor.shape, x.shape)

    def test_dask_array_input(self):
        x = da.zeros([3, 4])
        descriptor = Dataset.from_array(x, name='test')
        self.assertEqual(descriptor.shape, x.shape)

    def test_list_input(self):
        x = [[3, 4, 6], [5, 6, 7]]
        descriptor = Dataset.from_array(x, name='test')
        self.assertEqual(descriptor.shape, np.array(x).shape)

    def test_1d_main_data(self):
        values = np.ones([10])
        descriptor = Dataset.from_array(values)
        self.assertTrue(np.all([x == y for x, y in zip(values, descriptor)]))

        for dim in range(len(values.shape)):
            self.assertEqual(getattr(descriptor, string.ascii_lowercase[dim]),
                             getattr(descriptor, 'dim_{}'.format(dim)))

        self.assertFalse(hasattr(descriptor, 'dim_{}'.format(len(values.shape))))
        self.assertFalse(hasattr(descriptor, string.ascii_lowercase[len(values.shape)]))

    def test_2d_main_data(self):
        values = np.random.random([4, 5])
        descriptor = Dataset.from_array(values)

        for expected, actual in zip(values, descriptor):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))
        for dim in range(len(values.shape)):
            self.assertEqual(getattr(descriptor, string.ascii_lowercase[dim]),
                             getattr(descriptor, 'dim_{}'.format(dim)))

        self.assertFalse(hasattr(descriptor, 'dim_{}'.format(len(values.shape))))
        self.assertFalse(hasattr(descriptor, string.ascii_lowercase[len(values.shape)]))

    def test_3d_main_data(self):
        values = np.random.random([4, 5, 6])
        descriptor = Dataset.from_array(values)

        for expected, actual in zip(values, descriptor):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))

        for dim in range(len(values.shape)):
            self.assertEqual(getattr(descriptor, string.ascii_lowercase[dim]),
                             getattr(descriptor, 'dim_{}'.format(dim)))

        self.assertFalse(hasattr(descriptor, 'dim_{}'.format(len(values.shape))))
        self.assertFalse(hasattr(descriptor, string.ascii_lowercase[len(values.shape)]))

    def test_4d_main_data(self):
        values = np.random.random([4, 5, 7, 3])
        descriptor = Dataset.from_array(values)

        for expected, actual in zip(values, descriptor):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))
        for dim in range(len(values.shape)):
            self.assertEqual(getattr(descriptor, string.ascii_lowercase[dim]),
                             getattr(descriptor, 'dim_{}'.format(dim)))

        self.assertFalse(hasattr(descriptor, 'dim_{}'.format(len(values.shape))))
        self.assertFalse(hasattr(descriptor, string.ascii_lowercase[len(values.shape)]))

    def test_dimensions_not_matching_main(self):
        pass

    def test_unknown_data_type(self):
        values = np.random.random([4])
        descriptor = Dataset.from_array(values)

        expected = "Supported data_types for plotting are only:"
        with self.assertRaises(Warning) as context:
            descriptor.data_type = 'quark'
        self.assertTrue(expected in str(context.exception))

    def test_enum_data_type(self):
        values = np.random.random([4])
        descriptor = Dataset.from_array(values)
        for dt_type in DataTypes:
            descriptor.data_type = dt_type
            self.assertTrue(descriptor.data_type == dt_type)

    def test_string_data_type(self):
        values = np.random.random([4])
        descriptor = Dataset.from_array(values)
        for dt_type in DataTypes:
            descriptor.data_type = str(dt_type.name)
            self.assertTrue(descriptor.data_type == dt_type)


class TestDatasetRepr(unittest.TestCase):

    def test_minimal_inputs(self):
        values = np.arange(5)

        descriptor = Dataset.from_array(values)
        actual = '{}'.format(descriptor)

        out = 'generic'
        da_array = da.from_array(values, name=out)

        expected = 'sidpy Dataset of type {} with:\n '.format(DataTypes.UNKNOWN.name)
        expected = expected + '{}'.format(da_array)
        expected = expected + '\n data contains: {} ({})'.format(out, out)
        expected = expected + '\n and Dimensions: '
        expected = expected + '\n  {}:  {} ({}) of size {}'.format('a', out, out, values.shape)
        self.assertEqual(actual, expected)

    def test_fully_configured(self):
        values = np.arange(5)

        descriptor = Dataset.from_array(values)
        for att in generic_attributes:
            setattr(descriptor, att, 'test')
        descriptor.metadata = {0: 'test'}

        actual = '{}'.format(descriptor)

        out = 'test'
        da_array = da.from_array(values, name='generic')

        expected = 'sidpy Dataset of type {} with:\n '.format(DataTypes.UNKNOWN.name)
        expected = expected + '{}'.format(da_array)
        expected = expected + '\n data contains: {} ({})'.format(out, out)
        expected = expected + '\n and Dimensions: '
        expected = expected + '\n  {}:  {} ({}) of size {}'.format('a', 'generic', 'generic', values.shape)
        expected = expected + '\n with metadata: {}'.format([0])

        self.assertEqual(actual, expected)

    def test_user_defined_parameters(self):
        # self.blah = 14. Will / should this get printed
        pass


class TestLikeData(unittest.TestCase):

    def test_minimal_inputs(self):
        values = np.ones([4,5])
        source_dset = Dataset.from_array(values)
        values = np.zeros([4, 5])
        descriptor = source_dset.like_data(values)
        self.assertTrue(descriptor.shape ==values.shape)
        self.assertIsInstance(descriptor, Dataset)

    def test_all_customized_properties(self):
        values = np.ones([4,5])
        source_dset = Dataset.from_array(values)
        for att in generic_attributes:
            setattr(source_dset, att, 'test')
        source_dset.metadata = {0: 'test'}

        values = np.zeros([4, 5])
        descriptor = source_dset.like_data(values)

        self.assertEqual(descriptor.title, 'like test')
        descriptor.title = 'test'
        self.assertTrue(np.all([getattr(descriptor, att) == 'test' for att in generic_attributes]))

        self.assertEqual(descriptor.metadata, source_dset.metadata)
        self.assertEqual(descriptor.original_metadata, source_dset.original_metadata)

    def test_changing_size(self):
        values = np.ones([4, 5])
        source_dset = Dataset.from_array(values)
        source_dset.a *= 0.5
        source_dset.quantity = 'test'
        values = np.zeros([3, 5])
        descriptor = source_dset.like_data(values)

        self.assertEqual(descriptor.a, np.arange(3)*.5)


class TestCopy(unittest.TestCase):

    def test_minimal_inputs(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)

        # Just the main dataset
        pass

    def test_all_customized_properties(self):
        pass


class TestRenameDimension(unittest.TestCase):

    def test_valid_index_and_name(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        descriptor.rename_dimension(0, 'v')
        self.assertEqual(descriptor.v, descriptor.dim_0)


    def test_invalid_index_object_type(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        with self.assertRaises(ValueError):
            descriptor.rename_dimension('v', 'v')

    def test_index_out_of_bounds(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        with self.assertRaises(KeyError):
            descriptor.rename_dimension(3, 'v')

    def test_invalid_name_object_types(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        with self.assertRaises(ValueError):
            descriptor.rename_dimension(0, 1)

    def test_empty_name_string(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        with self.assertRaises(ValueError):
            descriptor.rename_dimension(0, '')

    def test_existing_name(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        with self.assertRaises(ValueError):
            descriptor.rename_dimension(0, 'b')


class TestSetDimension(unittest.TestCase):

    def test_valid_index_and_dim_obj(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        descriptor.set_dimension(0,Dimension(np.arange(4),'x', quantity='test', units='test' ))
        self.assertIsInstance(descriptor.x, Dimension)

    def test_invalid_dim_object(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)

        with self.assertRaises(ValueError):
            descriptor.set_dimension(3, Dimension(np.arange(4), 'x', quantity='test', units='test'))
        with self.assertRaises(ValueError):
            descriptor.set_dimension('2', Dimension(np.arange(4), 'x', quantity='test', units='test'))
        with self.assertRaises(ValueError):
            descriptor.set_dimension(2, np.arange(4))


    # validity of index tested in TestRenameDimension


class TestViewMetadata(unittest.TestCase):

    def test_default_empty_metadata(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        # self.assertEqual('{}'.format(descriptor.view_metadata()),'None')


    def test_entered_metadata(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        descriptor.metadata= {0: 'test'}

        # print('{}'.format(descriptor.view_metadata()))

        # self.assertEqual(descriptor.view_metadata(), '0 : test')




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
