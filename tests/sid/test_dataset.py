# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 17:07:16 2020

@author: Suhas Somnath, Gerd Duscher
"""
from __future__ import division, print_function, unicode_literals, \
    absolute_import
import unittest

import numpy as np
import dask.array as da
import string
import ase.build
import sys
from copy import deepcopy

sys.path.insert(0, "../../sidpy/")

from sidpy.sid.dimension import Dimension
from sidpy.sid.dataset import DataType, Dataset

if sys.version_info.major == 3:
    unicode = str

generic_attributes = ['title', 'quantity', 'units', 'modality', 'source']


def validate_dataset_properties(self, dataset, values,
                                title='generic', quantity='generic', units='generic',
                                modality='generic', source='generic', dimension_dict=None,
                                data_type=DataType.UNKNOWN, variance=None,
                                metadata={}, original_metadata={},
                                ):
    
    if 'fitting_functions' in dataset.metadata.keys():
        metadata['fitting_functions'] = dataset.metadata['fitting_functions']
        self.assertIsInstance(self, unittest.TestCase)
        self.assertIsInstance(dataset, Dataset)
        # DONE: Validate that EVERY property is set correctly
        values = np.array(values)

        self.assertTrue(np.all([hasattr(dataset, att) for att in generic_attributes]))

        expected = values.flatten()
        actual = dataset.compute().flatten()
        self.assertTrue(np.allclose(expected, actual, equal_nan=True, rtol=1e-05, atol=1e-08))
        # self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))

        this_attributes = [title, quantity, units, modality, source]
        dataset_attributes = [getattr(dataset, att) for att in generic_attributes]

        for expected, actual in zip(dataset_attributes, this_attributes):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))

        if variance is None:
            self.assertEqual(dataset.variance, None)
        else:
            self.assertTrue(isinstance(dataset.variance, da.core.Array))
            expected_var = np.array(variance).flatten()
            actual_var = dataset.variance.compute().flatten()
            self.assertTrue(np.allclose(expected_var, actual_var, equal_nan=True, rtol=1e-05, atol=1e-08))

        self.assertEqual(dataset.data_type, data_type)

        self.assertEqual(dataset.metadata, metadata)
        self.assertEqual(dataset.original_metadata, original_metadata)

        if dimension_dict is None:
            for dim in range(len(values.shape)):
                self.assertEqual(getattr(dataset, string.ascii_lowercase[dim]),
                                getattr(dataset, 'dim_{}'.format(dim)))
        else:
            for dim in range(len(values.shape)):
                self.assertEqual(getattr(dataset, dimension_dict[dim].name),
                                getattr(dataset, 'dim_{}'.format(dim)))
                self.assertEqual(dataset._axes[dim], dimension_dict[dim])

        # Make sure we do not have too many dimensions
        self.assertFalse(hasattr(dataset, 'dim_{}'.format(len(values.shape))))
        
    else:
        self.assertIsInstance(self, unittest.TestCase)
        self.assertIsInstance(dataset, Dataset)
        # DONE: Validate that EVERY property is set correctly
        values = np.array(values)

        self.assertTrue(np.all([hasattr(dataset, att) for att in generic_attributes]))

        expected = values.flatten()
        actual = dataset.compute().flatten()
        self.assertTrue(np.allclose(expected, actual, equal_nan=True, rtol=1e-05, atol=1e-08))
        # self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))

        this_attributes = [title, quantity, units, modality, source]
        dataset_attributes = [getattr(dataset, att) for att in generic_attributes]

        for expected, actual in zip(dataset_attributes, this_attributes):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))

        if variance is None:
            self.assertEqual(dataset.variance, None)
        else:
            self.assertTrue(isinstance(dataset.variance, da.core.Array))
            expected_var = np.array(variance).flatten()
            actual_var = dataset.variance.compute().flatten()
            self.assertTrue(np.allclose(expected_var, actual_var, equal_nan=True, rtol=1e-05, atol=1e-08))

        self.assertEqual(dataset.data_type, data_type)

        self.assertEqual(dataset.metadata, metadata)
        self.assertEqual(dataset.original_metadata, original_metadata)

        if dimension_dict is None:
            for dim in range(len(values.shape)):
                self.assertEqual(getattr(dataset, string.ascii_lowercase[dim]),
                                getattr(dataset, 'dim_{}'.format(dim)))
        else:
            for dim in range(len(values.shape)):
                self.assertEqual(getattr(dataset, dimension_dict[dim].name),
                                getattr(dataset, 'dim_{}'.format(dim)))
                self.assertEqual(dataset._axes[dim], dimension_dict[dim])

        # Make sure we do not have too many dimensions
        self.assertFalse(hasattr(dataset, 'dim_{}'.format(len(values.shape))))
        # self.assertFalse(hasattr(dataset, string.ascii_lowercase[len(values.shape)]))


# Following 4 methods are used in testing the methods that reduce dimensions of the dataset
def single_axis_test(self, func, **kwargs):
    dset_np = np.random.rand(4, 1, 5)
    dset = Dataset.from_array(dset_np, title='test')
    sid_func = getattr(dset, func)
    np_func = getattr(dset_np, func)
    dset_1 = sid_func(axis=0, keepdims=False)
    dim_dict = {0: dset._axes[1].copy(), 1: dset._axes[2].copy()}

    title_prefix = kwargs.get('title_prefix')
    validate_dataset_properties(self, dset_1, np_func(axis=0, keepdims=False),
                                title=title_prefix + dset.title,
                                modality=dset.modality, source=dset.modality, dimension_dict=dim_dict,
                                data_type=DataType.UNKNOWN,
                                metadata={}, original_metadata={}
                                )


def multiple_axes_test(self, func, **kwargs):
    dset_np = np.random.rand(1, 6, 4)
    dset = Dataset.from_array(dset_np, title='test')
    sid_func = getattr(dset, func)
    np_func = getattr(dset_np, func)
    dset_1 = sid_func(axis=(0, 1), keepdims=False)
    dim_dict = {0: dset._axes[2].copy()}

    title_prefix = kwargs.get('title_prefix')
    validate_dataset_properties(self, dset_1, np_func(axis=(0, 1), keepdims=False),
                                title=title_prefix + dset.title,
                                modality=dset.modality, source=dset.modality, dimension_dict=dim_dict,
                                data_type=DataType.UNKNOWN,
                                metadata={}, original_metadata={}
                                )


# The following two tests are for when keep_dims is set to True
def keepdims_test(self, func, **kwargs):
    dset_np = np.random.rand(2, 1, 4)
    dset = Dataset.from_array(dset_np, title='test')
    sid_func = getattr(dset, func)
    np_func = getattr(dset_np, func)

    dset_1 = sid_func(axis=0, keepdims=True)

    dim_dict = dset._axes.copy()
    dim_dict[0] = Dimension(np.arange(1), name=dset._axes[0].name,
                            quantity=dset._axes[0].quantity, units=dset._axes[0].units,
                            dimension_type=dset._axes[0].dimension_type)

    title_prefix = kwargs.get('title_prefix')
    validate_dataset_properties(self, dset_1, np_func(axis=0, keepdims=True),
                                title=title_prefix + dset.title,
                                modality=dset.modality, source=dset.modality, dimension_dict=dim_dict,
                                data_type=DataType.UNKNOWN,
                                metadata={}, original_metadata={}
                                )


def keepdims_multiple_axes_test(self, func, **kwargs):
    dset_np = np.random.rand(1, 5, 4)
    dset = Dataset.from_array(dset_np, title='test')
    sid_func = getattr(dset, func)
    np_func = getattr(dset_np, func)
    title_prefix = kwargs.get('title_prefix')

    dset_1 = sid_func(axis=(0, 1), keepdims=True)
    dim_dict = dset._axes.copy()
    dim_dict[0] = Dimension(np.arange(1), name=dset._axes[0].name,
                            quantity=dset._axes[0].quantity, units=dset._axes[0].units,
                            dimension_type=dset._axes[0].dimension_type)
    dim_dict[1] = Dimension(np.arange(1), name=dset._axes[1].name,
                            quantity=dset._axes[1].quantity, units=dset._axes[1].units,
                            dimension_type=dset._axes[1].dimension_type)
    validate_dataset_properties(self, dset_1, np_func(axis=(0, 1), keepdims=True),
                                title=title_prefix + dset.title,
                                modality=dset.modality, source=dset.modality, dimension_dict=dim_dict,
                                data_type=DataType.UNKNOWN,
                                metadata={}, original_metadata={}
                                )


class TestDatasetFromArray(unittest.TestCase):

    def test_std_inputs(self):
        # verify generic properties, dimensions, etc.
        values = np.random.random([4, 5, 6])
        descriptor = Dataset.from_array(values)

        validate_dataset_properties(self, descriptor, values)

    def test_dset_with_variance(self):
        values = np.random.random([4, 5, 6])
        variance = np.random.random([4, 5, 6])
        descriptor = Dataset.from_array(values, variance=variance)
        validate_dataset_properties(self, descriptor, values, variance=variance)


class TestDatasetConstructor(unittest.TestCase):

    def test_minimal_inputs(self):
        """ test minimum input requirement of an array like object
        """
        with self.assertRaises(TypeError):
            Dataset.from_array()
        descriptor = Dataset.from_array(np.arange(3))
        validate_dataset_properties(self, descriptor, np.arange(3))

    def test_all_inputs(self):
        descriptor = Dataset.from_array(np.arange(3), title='test')
        validate_dataset_properties(self, descriptor, np.arange(3), title='test')

    def test_user_defined_parms(self):
        descriptor = Dataset.from_array(np.arange(3), title='test')

        for att in generic_attributes:
            setattr(descriptor, att, 'test')

        test_dict = {0: 'test'}
        descriptor.metadata = test_dict.copy()
        descriptor.original_metadata = test_dict.copy()

        validate_dataset_properties(self, descriptor, np.arange(3),
                                    title='test', quantity='test', units='test',
                                    modality='test', source='test', dimension_dict=None,
                                    data_type=DataType.UNKNOWN,
                                    metadata=test_dict, original_metadata=test_dict
                                    )

    def test_invalid_main_types(self):
        """
        anything that is not recognized by dask will make an empty dask array
        but name has to be a string
        """
        # TODO: call validate_dataset_properties instead
        descriptor = Dataset.from_array(DataType.UNKNOWN)
        self.assertEqual(descriptor.shape, ())

        descriptor = Dataset.from_array('test')
        self.assertEqual(descriptor.shape, ())

        descriptor = Dataset.from_array(1)
        self.assertEqual(descriptor.shape, ())

        with self.assertRaises(ValueError):
            Dataset.from_array(1, 1)
        # TODO: Should be TypeError

    def test_numpy_array_input(self):
        x = np.ones([3, 4, 5])
        descriptor = Dataset.from_array(x, title='test')
        self.assertEqual(descriptor.shape, x.shape)
        # TODO: call validate_dataset_properties instead

    def test_dask_array_input(self):
        x = da.zeros([3, 4], chunks='auto')
        descriptor = Dataset.from_array(x, chunks='auto', title='test')
        self.assertEqual(descriptor.shape, x.shape)
        # TODO: call validate_dataset_properties instead

    def test_list_input(self):
        x = [[3, 4, 6], [5, 6, 7]]
        descriptor = Dataset.from_array(x, title='test')
        self.assertEqual(descriptor.shape, np.array(x).shape)
        # TODO: call validate_dataset_properties instead

    def test_1d_main_data(self):
        values = np.ones([10])
        descriptor = Dataset.from_array(values)
        self.assertTrue(np.all([x == y for x, y in zip(values, descriptor)]))

        # TODO: call validate_dataset_properties instead
        # Move such validation to validate_dataset_properties
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
        for dt_type in DataType:
            descriptor.data_type = dt_type
            self.assertTrue(descriptor.data_type == dt_type)

    def test_string_data_type(self):
        values = np.random.random([4])
        descriptor = Dataset.from_array(values)
        for dt_type in DataType:
            descriptor.data_type = str(dt_type.name)
            self.assertTrue(descriptor.data_type == dt_type)


class TestDatasetRepr(unittest.TestCase):

    def test_minimal_inputs(self):
        values = np.arange(5)

        descriptor = Dataset.from_array(values)
        actual = '{}'.format(descriptor)

        out = 'generic'
        da_array = da.from_array(values, chunks='auto')

        expected = 'sidpy.Dataset of type {} with:\n '.format(DataType.UNKNOWN.name)
        expected = expected + '{}'.format(da_array)
        expected = expected + '\n data contains: {} ({})'.format(out, out)
        expected = expected + '\n and Dimensions: '
        expected = expected + '\n{}:  {} ({}) of size {}'.format('a', out, out, values.shape)

        """
        for exp, act in zip(expected.split('\n'), actual.split('\n')):
            print('Expected:\t' + exp)
            print('Actual:\t' + act)
            print(exp == act)
        """

        self.assertEqual(actual, expected)

    def test_fully_configured(self):
        values = np.arange(5)

        descriptor = Dataset.from_array(values)
        for att in generic_attributes:
            setattr(descriptor, att, 'test')
        descriptor.metadata = {0: 'test'}

        actual = '{}'.format(descriptor)

        out = 'test'
        da_array = da.from_array(values, chunks='auto')

        expected = 'sidpy.Dataset of type {} with:\n '.format(DataType.UNKNOWN.name)
        expected = expected + '{}'.format(da_array)
        expected = expected + '\n data contains: {} ({})'.format(out, out)
        expected = expected + '\n and Dimensions: '
        expected = expected + '\n{}:  {} ({}) of size {}'.format('a', 'generic', 'generic', values.shape)
        expected = expected + '\n with metadata: {}'.format([0])

        """
        for exp, act in zip(expected.split('\n'), actual.split('\n')):
            print('Expected:\t' + exp)
            print('Actual:\t' + act)
            print(exp == act)
        """

        self.assertEqual(actual, expected)

    def test_user_defined_parameters(self):
        # self.blah = 14. Will / should this get printed
        pass


class TestLikeData(unittest.TestCase):

    def test_minimal_inputs(self):
        values = np.ones([4, 5])
        source_dset = Dataset.from_array(values)
        values = np.zeros([4, 5])
        descriptor = source_dset.like_data(values)
        self.assertTrue(descriptor.shape == values.shape)
        self.assertIsInstance(descriptor, Dataset)

    def test_all_customized_properties(self):
        values = np.ones([4, 5])
        source_dset = Dataset.from_array(values)
        for att in generic_attributes:
            setattr(source_dset, att, 'test')
        source_dset.metadata = {0: 'test'}

        values = np.zeros([4, 5])
        descriptor = source_dset.like_data(values)

        self.assertEqual(descriptor.title, 'test_new')
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

        # self.assertEqual(descriptor.a.values), np.arange(3)*.5)
        expected = descriptor.a.values
        actual = np.arange(3) * .5
        # self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))

    def test_variance(self):
        values = np.ones([4, 5])
        var = np.random.normal(size=(4, 5))
        source_dset = Dataset.from_array(values, variance=var)
        descriptor = source_dset.like_data(values)
        self.assertEqual(descriptor.variance, None)
        descriptor = source_dset.like_data(values, variance=var)
        self.assertEqual(descriptor.variance.all(), source_dset.variance.all())


class TestCopy(unittest.TestCase):

    def test_minimal_inputs(self):
        values = np.random.random([4, 5])
        dataset = Dataset.from_array(values)

        descriptor = dataset.copy()

        self.assertIsInstance(descriptor, Dataset)
        for expected, actual in zip(dataset, descriptor):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))

        self.assertTrue(np.all([hasattr(descriptor, att) for att in generic_attributes]))

        self.assertTrue(np.all([getattr(descriptor, att) == 'generic' for att in generic_attributes]))

        self.assertEqual(descriptor.data_type, DataType.UNKNOWN)

        self.assertEqual(descriptor.metadata, {})
        self.assertEqual(descriptor.original_metadata, {})

        for dim in range(len(values.shape)):
            self.assertEqual(getattr(descriptor, string.ascii_lowercase[dim]),
                             getattr(descriptor, 'dim_{}'.format(dim)))

        self.assertFalse(hasattr(descriptor, 'dim_{}'.format(len(dataset.shape))))
        self.assertFalse(hasattr(descriptor, string.ascii_lowercase[len(dataset.shape)]))

    def test_all_customized_properties(self):
        values = np.random.random([4, 5])
        dataset = Dataset.from_array(values)
        dataset.rename_dimension(0, 'x')
        dataset.quantity = 'test'
        descriptor = dataset.copy()

        self.assertIsInstance(descriptor, Dataset)
        self.assertEqual(descriptor.quantity, dataset.quantity)
        self.assertTrue(hasattr(descriptor, 'x'))


class TestRenameDimension(unittest.TestCase):

    def test_valid_index_and_name(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        descriptor.rename_dimension(0, 'v')
        self.assertEqual(descriptor.v, descriptor.dim_0)

    def test_invalid_index_object_type(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        with self.assertRaises(TypeError):
            descriptor.rename_dimension('v', 'v')

    def test_index_out_of_bounds(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        with self.assertRaises(IndexError):
            descriptor.rename_dimension(3, 'v')

    def test_invalid_name_object_types(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        with self.assertRaises(TypeError):
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
        descriptor.set_dimension(0, Dimension(np.arange(4), 'x', quantity='test', units='test'))
        self.assertIsInstance(descriptor.x, Dimension)

    def test_invalid_dim_object(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)

        with self.assertRaises(TypeError):
            descriptor.set_dimension(3, "New dimension")
        with self.assertRaises(TypeError):
            descriptor.set_dimension('2', {'x': np.arange(4)})
        with self.assertRaises(TypeError):
            descriptor.set_dimension(2, np.arange(4))

    def test_dim_obj_slope(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        descriptor.set_dimension(0, Dimension(np.arange(4), 'x', quantity='test', units='test'))
        self.assertTrue(descriptor.x.slope==1)

    # validity of index tested in TestRenameDimension


class TestHelperFunctions(unittest.TestCase):
    def test_get_image_dims(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        descriptor.set_dimension(0, Dimension(np.arange(4), 'x', quantity='test', dimension_type='spatial'))

        image_dims = descriptor.get_image_dims()
        self.assertEqual(len(image_dims), 1)
        self.assertEqual(image_dims[0], 0)

        descriptor.dim_1.dimension_type = 'spatial'
        image_dims = descriptor.get_image_dims()
        self.assertEqual(len(image_dims), 2)
        self.assertEqual(image_dims[1], 1)

    def test_get_dimensions_by_type(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        descriptor.set_dimension(0, Dimension(np.arange(4), 'x', quantity='test', dimension_type='spatial'))

        image_dims = descriptor.get_dimensions_by_type('spatial')
        self.assertEqual(len(image_dims), 1)
        self.assertEqual(image_dims[0], 0)

        descriptor.dim_1.dimension_type = 'spatial'
        image_dims = descriptor.get_dimensions_by_type('spatial')
        self.assertEqual(len(image_dims), 2)
        self.assertEqual(image_dims[1], 1)

    def test_get_spectral_dims(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        descriptor.set_dimension(0, Dimension(np.arange(4), 'x', quantity='test', dimension_type='spatial'))

        spec_dims = descriptor.get_spectral_dims()
        self.assertEqual(len(spec_dims), 0)
        descriptor.x.dimension_type = 'spectral'
        spec_dims = descriptor.get_spectral_dims()
        self.assertEqual(len(spec_dims), 1)
        self.assertEqual(spec_dims[0], 0)

        descriptor.dim_1.dimension_type = 'spectral'
        spec_dims = descriptor.get_spectral_dims()
        self.assertEqual(len(spec_dims), 2)
        self.assertEqual(spec_dims[1], 1)

    def test_get_extent(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        descriptor.set_dimension(0, Dimension(np.arange(4), 'x', quantity='test', dimension_type='spatial'))
        descriptor.dim_1.dimension_type = 'spatial'
        descriptor.set_dimension(0, Dimension(np.arange(4), 'x', quantity='test', dimension_type='spatial'))

        extent = descriptor.get_extent([0, 1])
        self.assertEqual(extent[0], -0.5)
        self.assertEqual(extent[1], 3.5)

    def test_get_labels(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        labels = descriptor.labels
        self.assertEqual(labels[0], 'generic (generic)')

    def test_empty_structure(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        structures = descriptor.structures
        self.assertEqual(len(structures), 0)

    def test_add_structure(self):
        values = np.zeros([4, 5])
        a = 5.14  # A
        atoms = ase.build.bulk('Si', 'diamond', a=a, cubic=True)
        descriptor = Dataset.from_array(values)
        descriptor.add_structure(atoms)
        descriptor.add_structure(atoms, 'reference')

        self.assertEqual(len(descriptor.structures), 2)
        self.assertTrue('reference' in descriptor.structures.keys())

    def test__equ__(self):
        values = np.zeros([4, 5])
        descriptor1 = Dataset.from_array(values)
        descriptor2 = Dataset.from_array(values)
        # TODO: why does direct comparison not work
        self.assertTrue(descriptor1.__eq__(descriptor2))
        self.assertFalse(descriptor1.__eq__(np.arange(4)))

        descriptor1.set_dimension(0, Dimension(np.arange(4), 'x', quantity='test', dimension_type='spatial'))
        self.assertFalse(descriptor1.__eq__(descriptor2))

        descriptor2.modality = 'nix'
        self.assertFalse(descriptor1.__eq__(descriptor2))

        descriptor2.data_type = 'image'
        self.assertFalse(descriptor1.__eq__(descriptor2))

        descriptor2.source = 'image'
        self.assertFalse(descriptor1.__eq__(descriptor2))

        descriptor2.quantity = 'image'
        self.assertFalse(descriptor1.__eq__(descriptor2))
        descriptor2.units = 'image'
        self.assertFalse(descriptor1.__eq__(descriptor2))

    def test_h5_dataset(self):
        values = np.ones([4, 5])
        source_dset = Dataset.from_array(values)

    def test_auto_provenance(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        self.assertEqual(list(descriptor.provenance.keys())[0][:5] ,  'sidpy' )
        summed_dataset = descriptor.sum(axis=0)
        self.assertEqual(list(descriptor.provenance.keys())[0][:5] ,  'sidpy' )
        self.assertEqual(list(summed_dataset.provenance.keys())[1][:3] ,  'sum' )

    def test_add_Prevenance(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        descriptor.add_provenance('test', 'test_function', version=1)
        self.assertEqual(list(descriptor.provenance.keys())[1][:4], 'test')

class TestViewMetadata(unittest.TestCase):

    def test_default_empty_metadata(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        descriptor.view_metadata()
        # self.assertEqual('{}'.format(descriptor.view_metadata()),'None')

    def test_entered_metadata(self):
        values = np.zeros([4, 5])
        descriptor = Dataset.from_array(values)
        descriptor.metadata = {0: 'test'}

        print('{}'.format(descriptor.view_metadata()))

        # self.assertEqual(descriptor.view_metadata(), '0 : test')


class TestViewOriginalMetadata(unittest.TestCase):

    def test_default_empty_metadata(self):
        pass

    def test_entered_metadata(self):
        pass


class Testallmethod(unittest.TestCase):
    def test_all_single_axis(self):
        single_axis_test(self, 'all', title_prefix='all_aggregate_')

    def test_all_multiple_axes(self):
        multiple_axes_test(self, 'all', title_prefix='all_aggregate_')

    def test_all_keepdims(self):
        keepdims_test(self, 'all', title_prefix='all_aggregate_')

    def test_all_keepdims_multiple_axes(self):
        keepdims_multiple_axes_test(self, 'all', title_prefix='all_aggregate_')


class Testanymethod(unittest.TestCase):
    def test_any_single_axis(self):
        single_axis_test(self, 'any', title_prefix='any_aggregate_')

    def test_any_multiple_axes(self):
        multiple_axes_test(self, 'any', title_prefix='any_aggregate_')

    def test_any_keepdims(self):
        keepdims_test(self, 'any', title_prefix='any_aggregate_')

    def test_any_keepdims_multiple_axes(self):
        keepdims_multiple_axes_test(self, 'any', title_prefix='any_aggregate_')


class TestMinMethod(unittest.TestCase):
    def test_min_single_axis(self):
        single_axis_test(self, 'min', title_prefix='min_aggregate_')

    def test_min_multiple_axes(self):
        multiple_axes_test(self, 'min', title_prefix='min_aggregate_')

    def test_min_keepdims(self):
        keepdims_test(self, 'min', title_prefix='min_aggregate_')

    def test_min_keepdims_multiple_axes(self):
        keepdims_multiple_axes_test(self, 'min', title_prefix='min_aggregate_')


class TestMaxMethod(unittest.TestCase):
    def test_max_single_axis(self):
        single_axis_test(self, 'max', title_prefix='max_aggregate_')

    def test_max_multiple_axes(self):
        multiple_axes_test(self, 'max', title_prefix='max_aggregate_')

    def test_max_keepdims(self):
        keepdims_test(self, 'max', title_prefix='max_aggregate_')

    def test_min_keepdims_multiple_axes(self):
        keepdims_multiple_axes_test(self, 'max', title_prefix='max_aggregate_')


class TestSumMethod(unittest.TestCase):
    def test_sum_single_axis(self):
        single_axis_test(self, 'sum', title_prefix='sum_aggregate_')

    def test_sum_multiple_axis(self):
        multiple_axes_test(self, 'sum', title_prefix='sum_aggregate_')

    def test_sum_keepdims(self):
        keepdims_test(self, 'sum', title_prefix='sum_aggregate_')

    def test_sum_keepdims_multiple_axis(self):
        keepdims_multiple_axes_test(self, 'sum', title_prefix='sum_aggregate_')

    def test_sum_dtype(self):
        # Have to take care of complex datasets when asked about the sum of the entire dataset
        pass


class TestMeanMethod(unittest.TestCase):
    def test_mean_single_axis(self):
        single_axis_test(self, 'mean', title_prefix='mean_aggregate_')

    def test_mean_multiple_axis(self):
        multiple_axes_test(self, 'mean', title_prefix='mean_aggregate_')

    def test_mean_keepdims(self):
        keepdims_test(self, 'mean', title_prefix='mean_aggregate_')

    def test_mean_keepdims_multiple_axis(self):
        keepdims_multiple_axes_test(self, 'mean', title_prefix='mean_aggregate_')

    def test_mean_dtype(self):
        # Have to take care of complex datasets when asked about the sum of the entire dataset
        pass


class TestSlicing(unittest.TestCase):
    np.random.seed(0)
    values = np.random.rand(3, 4, 6, 5)
    dset = Dataset.from_array(values, title='4D_STEM', units='nA',
                              quantity='Current',
                              modality='modality', source='source')
    dset.data_type = DataType.IMAGE_4D
    dset.metadata = {'info_1': np.linspace(0, 5.6, 30), 'instrument': 'opportunity rover AFM'}

    x_dim = np.linspace(0, 1E-6,
                        dset.shape[0])
    y_dim = np.linspace(0, 2E-6,
                        dset.shape[1])
    kx_dim = np.linspace(0, 12, dset.shape[2])
    ky_dim = np.linspace(0, 10, dset.shape[3])

    dset.set_dimension(0, Dimension(x_dim,
                                    name='x',
                                    units='m', quantity='x',
                                    dimension_type='spatial'))
    dset.set_dimension(1, Dimension(y_dim,
                                    name='y',
                                    units='m', quantity='y',
                                    dimension_type='spatial'))
    dset.set_dimension(2, Dimension(kx_dim,
                                    name='Intensity KX',
                                    units='counts', quantity='Intensity',
                                    dimension_type='spectral'))
    dset.set_dimension(3, Dimension(ky_dim,
                                    name='Intensity KY',
                                    units='counts', quantity='Intensity',
                                    dimension_type='spectral'))

    def test_getitem_integer(self):
        # Create a sample Dask array
        old_dset = self.dset
        sliced = self.dset[:, 2]
        dim_dict = {0: old_dset._axes[0].copy(),
                    1: old_dset._axes[2].copy(),
                    2: old_dset._axes[3].copy()}

        validate_dataset_properties(self, sliced, self.dset.compute()[:, 2],
                                    title=self.dset.title, quantity=self.dset.quantity,
                                    units=self.dset.units,
                                    modality=self.dset.modality, source=self.dset.source,
                                    dimension_dict=dim_dict,
                                    data_type=self.dset.data_type,
                                    metadata=self.dset.metadata, original_metadata=self.dset.original_metadata)

    def test_getitem_NoneandEllipsis1(self):
        old_dset = self.dset
        sliced = self.dset[..., None, :]

        dim_dict = {0: old_dset._axes[0].copy(),
                    1: old_dset._axes[1].copy(),
                    2: old_dset._axes[2].copy(),
                    3: Dimension(1),
                    4: old_dset._axes[3].copy()}

        validate_dataset_properties(self, sliced, self.dset.compute()[..., None, :],
                                    title=self.dset.title, quantity=self.dset.quantity,
                                    units=self.dset.units,
                                    modality=self.dset.modality, source=self.dset.source,
                                    dimension_dict=dim_dict,
                                    data_type=self.dset.data_type,
                                    metadata=self.dset.metadata, original_metadata=self.dset.original_metadata)

    def test_getitem_NoneandEllipsis2(self):
        old_dset = self.dset

        sliced = self.dset[None, ..., None]
        dim_dict = {0: Dimension(1),
                    1: old_dset._axes[0].copy(),
                    2: old_dset._axes[1].copy(),
                    3: old_dset._axes[2].copy(),
                    4: old_dset._axes[3].copy(),
                    5: Dimension(1)}

        validate_dataset_properties(self, sliced, self.dset.compute()[None, ..., None],
                                    title=self.dset.title, quantity=self.dset.quantity,
                                    units=self.dset.units,
                                    modality=self.dset.modality, source=self.dset.source,
                                    dimension_dict=dim_dict,
                                    data_type=self.dset.data_type,
                                    metadata=self.dset.metadata, original_metadata=self.dset.original_metadata)

    def test_getitem_slice1(self):
        old_dset = self.dset
        sliced = self.dset[0:1]

        dim_dict = {0: deepcopy(old_dset._axes[0][0:1]),
                    1: deepcopy(old_dset._axes[1]),
                    2: deepcopy(old_dset._axes[2]),
                    3: deepcopy(old_dset._axes[3])}

        validate_dataset_properties(self, sliced, self.dset.compute()[0:1],
                                    title=self.dset.title, quantity=self.dset.quantity,
                                    units=self.dset.units,
                                    modality=self.dset.modality, source=self.dset.source,
                                    dimension_dict=dim_dict,
                                    data_type=self.dset.data_type,
                                    metadata=self.dset.metadata, original_metadata=self.dset.original_metadata)

    def test_getitem_slice2(self):
        old_dset = self.dset
        sliced = self.dset[0:3]

        dim_dict = {0: deepcopy(old_dset._axes[0][0:3]),
                    1: deepcopy(old_dset._axes[1]),
                    2: deepcopy(old_dset._axes[2]),
                    3: deepcopy(old_dset._axes[3])}

        validate_dataset_properties(self, sliced, self.dset.compute()[0:3],
                                    title=self.dset.title, quantity=self.dset.quantity,
                                    units=self.dset.units,
                                    modality=self.dset.modality, source=self.dset.source,
                                    dimension_dict=dim_dict,
                                    data_type=self.dset.data_type,
                                    metadata=self.dset.metadata, original_metadata=self.dset.original_metadata)

    def test_getitem_nparray(self):

        old_dset = self.dset
        inds = np.array([True, False, True, False, True, False])
        sliced = old_dset[:, :, inds, :]

        dim_dict = {0: deepcopy(old_dset._axes[0]),
                    1: deepcopy(old_dset._axes[1]),
                    2: deepcopy(old_dset._axes[2])[inds],
                    3: deepcopy(old_dset._axes[3])}

        validate_dataset_properties(self, sliced, self.dset.compute()[:, :, inds, :],
                                    title=self.dset.title, quantity=self.dset.quantity,
                                    units=self.dset.units,
                                    modality=self.dset.modality, source=self.dset.source,
                                    dimension_dict=dim_dict,
                                    data_type=self.dset.data_type,
                                    metadata=self.dset.metadata, original_metadata=self.dset.original_metadata)

    def test_getitem_daarray(self):
        np.random.seed(0)
        old_dset = self.dset
        inds = da.array(np.array([True, False, True, False, True]))
        sliced = old_dset[..., inds]

        dim_dict = {0: deepcopy(old_dset._axes[0]),
                    1: deepcopy(old_dset._axes[1]),
                    2: deepcopy(old_dset._axes[2]),
                    3: deepcopy(old_dset._axes[3])[np.array(inds)]}

        validate_dataset_properties(self, sliced, self.dset.compute()[..., inds],
                                    title=self.dset.title, quantity=self.dset.quantity,
                                    units=self.dset.units,
                                    modality=self.dset.modality, source=self.dset.source,
                                    dimension_dict=dim_dict,
                                    data_type=self.dset.data_type,
                                    metadata=self.dset.metadata, original_metadata=self.dset.original_metadata)


if __name__ == '__main__':
    unittest.main()
