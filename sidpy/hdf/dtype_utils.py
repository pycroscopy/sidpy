# -*- coding: utf-8 -*-
"""
Utilities for transforming and validating data types

Given that many of the data transformations involve copying the data, they should
ideally happen in a lazy manner to avoid memory issues.

Created on Tue Nov  3 21:14:25 2015

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, absolute_import, unicode_literals, print_function
import sys
from warnings import warn
import h5py
import numpy as np
import dask.array as da

__all__ = ['flatten_complex_to_real', 'get_compound_sub_dtypes', 'flatten_compound_to_real', 'check_dtype',
           'stack_real_to_complex', 'validate_dtype', 'is_complex_dtype',
           'stack_real_to_compound', 'stack_real_to_target_dtype', 'flatten_to_real']

from sidpy.hdf.hdf_utils import lazy_load_array

if sys.version_info.major == 3:
    unicode = str


def flatten_complex_to_real(dataset, lazy=False):
    """
    Stacks the real values followed by the imaginary values in the last dimension of the given N dimensional matrix.
    Thus a complex matrix of shape (2, 3, 5) will turn into a matrix of shape (2, 3, 10)

    Parameters
    ----------
    dataset : array-like or :class:`numpy.ndarray`, or :class:`h5py.Dataset`, or :class:`dask.array.core.Array`
        Dataset of complex data type
    lazy : bool, optional. Default = False
        If set to True, will use lazy Dask arrays instead of in-memory numpy arrays

    Returns
    -------
    retval : :class:`numpy.ndarray`, or :class:`dask.array.core.Array`
        real valued dataset

    Examples
    --------
    >>> import numpy as np
    >>> import sidpy
    >>> length = 3
    >>> complex_array = np.random.randint(-5, high=5, size=length) + 1j * np.random.randint(-5, high=5, size=length)
    >>> print('Complex value: {} has shape: {}'.format(complex_array, complex_array.shape))
    Complex value: [2.-2.j 0.-3.j 0.-4.j] has shape: (3,)

    >>> stacked_real_array = sidpy.dtype_utils.flatten_complex_to_real(complex_array)
    >>> print('Stacked real value: {} has shape: '
    >>>       '{}'.format(stacked_real_array, stacked_real_array.shape))
    Stacked real value: [ 2.  0.  0. -2. -3. -4.] has shape: (6,)
    """
    if not isinstance(dataset, (h5py.Dataset, np.ndarray, da.core.Array)):
        raise TypeError('dataset should either be a h5py.Dataset or numpy / dask array')
    if not is_complex_dtype(dataset.dtype):
        raise TypeError("Expected a complex valued dataset")

    if isinstance(dataset, da.core.Array):
        lazy = True

    xp = np
    if lazy:
        dataset = lazy_load_array(dataset)
        xp = da

    axis = xp.array(dataset).ndim - 1
    if axis == -1:
        return xp.hstack([xp.real(dataset), xp.imag(dataset)])
    else:  # along the last axis
        return xp.concatenate([xp.real(dataset), xp.imag(dataset)], axis=axis)


def flatten_compound_to_real(dataset, lazy=False):
    """
    Flattens the individual components in a structured array or compound valued hdf5 dataset along the last axis to form
    a real valued array. Thus a compound h5py.Dataset or structured numpy matrix of shape (2, 3, 5) having 3 components
    will turn into a real valued matrix of shape (2, 3, 15), assuming that all the sub-dtypes of the matrix are real
    valued. ie - this function does not handle structured dtypes having complex values


    Parameters
    ----------
    dataset : :class:`numpy.ndarray`, or :class:`h5py.Dataset`, or :class:`dask.array.core.Array`
        Numpy array that is a structured array or a :class:`h5py.Dataset` of compound dtype
    lazy : bool, optional. Default = False
        If set to True, will use lazy Dask arrays instead of in-memory numpy arrays

    Returns
    -------
    retval : :class:`numpy.ndarray`, or :class:`dask.array.core.Array`
        real valued dataset

    Examples
    --------
    >>> import numpy as np
    >>> import sidpy
    >>> num_elems = 5
    >>> struct_dtype = np.dtype({'names': ['r', 'g', 'b'],
    >>>                          'formats': [np.float32, np.uint16, np.float64]})
    >>> structured_array = np.zeros(shape=num_elems, dtype=struct_dtype)
    >>> structured_array['r'] = np.random.random(size=num_elems) * 1024
    >>> structured_array['g'] = np.random.randint(0, high=1024, size=num_elems)
    >>> structured_array['b'] = np.random.random(size=num_elems) * 1024

    >>> print('Structured array is of shape {} and have values:'.format(structured_array.shape))
    >>> print(structured_array)
    Structured array is of shape (5,) and have values:
    [(859.62445,  54, 1012.22256219) (959.5565 , 678,  296.19788769)
     (383.20737, 689,  192.45427816) (201.56635, 889,  939.01082338)
     (334.22015, 467,  980.9081472 )]

    >>> real_array = sidpy.dtype_utils.flatten_compound_to_real(structured_array)
    >>> print("This array converted to regular scalar matrix has shape: {} and values:".format(real_array.shape))
    >>> print(real_array)
    This array converted to regular scalar matrix has shape: (15,) and values:
    [ 859.62445068  959.55651855  383.20736694  201.56634521  334.22015381
       54.          678.          689.          889.          467.
     1012.22256219  296.19788769  192.45427816  939.01082338  980.9081472 ]
    """
    if isinstance(dataset, h5py.Dataset):
        if len(dataset.dtype) == 0:
            raise TypeError("Expected compound h5py dataset")

        if lazy:
            xp = da
            dataset = lazy_load_array(dataset)
        else:
            xp = np
            warn('HDF5 datasets will be loaded as Dask arrays in the future. ie - kwarg lazy will default to True in future releases of sidpy')

        return xp.concatenate([xp.array(dataset[name]) for name in dataset.dtype.names], axis=len(dataset.shape) - 1)

    elif isinstance(dataset, (np.ndarray, da.core.Array)):
        if isinstance(dataset, da.core.Array):
            lazy = True

        xp = np
        if lazy:
            dataset = lazy_load_array(dataset)
            xp = da

        if len(dataset.dtype) == 0:
            raise TypeError("Expected structured array")
        if dataset.ndim > 0:
            return xp.concatenate([dataset[name] for name in dataset.dtype.names], axis=dataset.ndim - 1)
        else:
            return xp.hstack([dataset[name] for name in dataset.dtype.names])
    elif isinstance(dataset, np.void):
        return np.hstack([dataset[name] for name in dataset.dtype.names])
    else:
        raise TypeError('Datatype {} not supported'.format(type(dataset)))


def flatten_to_real(ds_main, lazy=False):
    """
    Flattens complex / compound / real valued arrays to real valued arrays

    Parameters
    ----------
    ds_main : :class:`numpy.ndarray`, or :class:`h5py.Dataset`, or :class:`dask.array.core.Array`
        Compound, complex or real valued numpy array or HDF5 dataset
    lazy : bool, optional. Default = False
        If set to True, will use lazy Dask arrays instead of in-memory numpy arrays

    Returns
    ----------
    ds_main : :class:`numpy.ndarray`, or :class:`dask.array.core.Array`
        Array raveled to a float data type

    Examples
    --------
    >>> import numpy as np
    >>> import sidpy
    >>> num_elems = 5
    >>> struct_dtype = np.dtype({'names': ['r', 'g', 'b'],
    >>>                          'formats': [np.float32, np.uint16, np.float64]})
    >>> structured_array = np.zeros(shape=num_elems, dtype=struct_dtype)
    >>> structured_array['r'] = np.random.random(size=num_elems) * 1024
    >>> structured_array['g'] = np.random.randint(0, high=1024, size=num_elems)
    >>> structured_array['b'] = np.random.random(size=num_elems) * 1024
    >>> print('Structured array is of shape {} and have values:'.format(structured_array.shape))
    >>> print(structured_array)
    Structured array is of shape (5,) and have values:
    [(859.62445,  54, 1012.22256219) (959.5565 , 678,  296.19788769)
     (383.20737, 689,  192.45427816) (201.56635, 889,  939.01082338)
     (334.22015, 467,  980.9081472 )]

    >>> real_array = sidpy.dtype_utils.flatten_to_real(structured_array)
    >>> print('This array converted to regular scalar matrix has shape: {} and values:'.format(real_array.shape))
    >>> print(real_array)
    This array converted to regular scalar matrix has shape: (15,) and values:
    [ 859.62445068  959.55651855  383.20736694  201.56634521  334.22015381
       54.          678.          689.          889.          467.
     1012.22256219  296.19788769  192.45427816  939.01082338  980.9081472 ]
    """
    if not isinstance(ds_main, (h5py.Dataset, np.ndarray, da.core.Array)):
        ds_main = np.array(ds_main)
    if is_complex_dtype(ds_main.dtype):
        return flatten_complex_to_real(ds_main, lazy=lazy)
    elif len(ds_main.dtype) > 0:
        return flatten_compound_to_real(ds_main, lazy=lazy)
    else:
        return ds_main


def get_compound_sub_dtypes(struct_dtype):
    """
    Returns a dictionary of the dtypes of each of the fields in the given structured array dtype

    Parameters
    ----------
    struct_dtype : :class:`numpy.dtype`
        dtype of a structured array

    Returns
    -------
    dtypes : dict
        Dictionary whose keys are the field names and values are the corresponding dtypes

    Examples
    --------
    >>> import numpy as np
    >>> import sidpy
    >>> struct_dtype = np.dtype({'names': ['r', 'g', 'b'],
    >>>                      'formats': [np.float32, np.uint16, np.float64]})
    >>> sub_dtypes = sidpy.dtype_utils.get_compound_sub_dtypes(struct_dtype)
    >>> for key, val in sub_dtypes.items():
    >>>     print('{} : {}'.format(key, val))
    g : uint16
    r : float32
    b : float64
    """
    if not isinstance(struct_dtype, np.dtype):
        raise TypeError('Provided object must be a structured array dtype')
    dtypes = dict()
    for field_name in struct_dtype.fields:
        dtypes[field_name] = struct_dtype.fields[field_name][0]
    return dtypes


def check_dtype(h5_dset):
    """
    Checks the datatype of the input HDF5 dataset and provides the appropriate
    function calls to convert it to a float

    Parameters
    ----------
    h5_dset : :class:`h5py.Dataset`
        Dataset of interest

    Returns
    -------
    func : callable
        function that will convert the dataset to a float
    is_complex : bool
        is the input dataset complex?
    is_compound : bool
        is the input dataset compound?
    n_features : Unsigned int
        Unsigned integer - the length of the 2nd dimension of the data after `func` is called on it
    type_mult : Unsigned int
        multiplier that converts from the typesize of the input :class:`~numpy.dtype` to the
        typesize of the data after func is run on it

    Examples
    --------
    >>> import numpy as np
    >>> import h5py
    >>> import sidpy
    >>> struct_dtype = np.dtype({'names': ['r', 'g', 'b'],
    >>>                      'formats': [np.float32, np.uint16, np.float64]})
    >>> file_path = 'dtype_utils_example.h5'
    >>> if os.path.exists(file_path):
    >>>     os.remove(file_path)
    >>> with h5py.File(file_path, mode='w') as h5_f:
    >>>     num_elems = (5, 7)
    >>>     structured_array = np.zeros(shape=num_elems, dtype=struct_dtype)
    >>>     structured_array['r'] = 450 * np.random.random(size=num_elems)
    >>>     structured_array['g'] = np.random.randint(0, high=1024, size=num_elems)
    >>>     structured_array['b'] = 3178 * np.random.random(size=num_elems)
    >>>     _ = h5_f.create_dataset('compound', data=structured_array)
    >>>     _ = h5_f.create_dataset('real', data=450 * np.random.random(size=num_elems), dtype=np.float16)
    >>>     _ = h5_f.create_dataset('complex', data=np.random.random(size=num_elems) + 1j * np.random.random(size=num_elems),
    >>>                             dtype=np.complex64)
    >>> h5_f.flush()
    >>> # Now, lets test the the function on compound-, complex-, and real-valued HDF5 datasets:
    >>> def check_dataset(h5_dset):
    >>>     print('\tDataset being tested: {}'.format(h5_dset))
    >>>     func, is_complex, is_compound, n_features, type_mult = sidpy.dtype_utils.check_dtype(h5_dset)
    >>>     print('\tFunction to transform to real: %s' % func)
    >>>     print('\tis_complex? %s' % is_complex)
    >>>     print('\tis_compound? %s' % is_compound)
    >>>     print('\tShape of dataset in its current form: {}'.format(h5_dset.shape))
    >>>     print('\tAfter flattening to real, shape is expected to be: ({}, {})'.format(h5_dset.shape[0], n_features))
    >>>     print('\tByte-size of a single element in its current form: {}'.format(type_mult))
    >>> with h5py.File(file_path, mode='r') as h5_f:
    >>>     print('Checking a compound-valued dataset:')
    >>>     check_dataset(h5_f['compound'])
    >>>     print('')
    >>>     print('Checking a complex-valued dataset:')
    >>>     check_dataset(h5_f['complex'])
    >>>    print('')
    >>>     print('Checking a real-valued dataset:')
    >>>     check_dataset(h5_f['real'])
    >>> os.remove(file_path)
    Checking a compound-valued dataset:
    Dataset being tested: <HDF5 dataset "compound": shape (5, 7), type "|V14">
    Function to transform to real: <function flatten_compound_to_real at 0x112c130d0>
    is_complex? False
    is_compound? True
    Shape of dataset in its current form: (5, 7)
    After flattening to real, shape is expected to be: (5, 21)
    Byte-size of a single element in its current form: 12
    - - - - - - - - - - - - - - - - - -
    Checking a complex-valued dataset:
    Dataset being tested: <HDF5 dataset "complex": shape (5, 7), type "<c8">
    Function to transform to real: <function flatten_complex_to_real at 0x112c13048>
    is_complex? True
    is_compound? False
    Shape of dataset in its current form: (5, 7)
    After flattening to real, shape is expected to be: (5, 14)
    Byte-size of a single element in its current form: 8
    - - - - - - - - - - - - - - - - - -
    Checking a real-valued dataset:
    Dataset being tested: <HDF5 dataset "real": shape (5, 7), type "<f2">
    Function to transform to real: <class 'numpy.float32'>
    is_complex? False
    is_compound? False
    Shape of dataset in its current form: (5, 7)
    After flattening to real, shape is expected to be: (5, 7)
    Byte-size of a single element in its current form: 4
    """
    if not isinstance(h5_dset, h5py.Dataset):
        raise TypeError('h5_dset should be a h5py.Dataset object')
    is_complex = False
    is_compound = False
    in_dtype = h5_dset.dtype
    # TODO: avoid assuming 2d shape - why does one even need n_samples!? We only care about the last dimension!
    n_features = h5_dset.shape[-1]
    if is_complex_dtype(h5_dset.dtype):
        is_complex = True
        new_dtype = np.real(h5_dset[0, 0]).dtype
        type_mult = new_dtype.itemsize * 2
        func = flatten_complex_to_real
        n_features *= 2
    elif len(h5_dset.dtype) > 0:
        """
        Some form of structured numpy is in use
        We only support real scalars for the component types at the current time
        """
        is_compound = True
        # TODO: Avoid hard-coding to float32
        new_dtype = np.float32
        type_mult = len(in_dtype) * new_dtype(0).itemsize
        func = flatten_compound_to_real
        n_features *= len(in_dtype)
    else:
        if h5_dset.dtype not in [np.float32, np.float64]:
            new_dtype = np.float32
        else:
            new_dtype = h5_dset.dtype.type

        type_mult = new_dtype(0).itemsize

        func = new_dtype

    return func, is_complex, is_compound, n_features, type_mult


def stack_real_to_complex(ds_real, lazy=False):
    """
    Puts the real and imaginary sections of the provided matrix (in the last axis) together to make complex matrix

    Parameters
    ------------
    ds_real : :class:`numpy.ndarray`, :class:`dask.array.core.Array`, or :class:`h5py.Dataset`
        n dimensional real-valued numpy array or HDF5 dataset where data arranged as [instance, 2 x features],
        where the first half of the features are the real component and the
        second half contains the imaginary components
    lazy : bool, optional. Default = False
        If set to True, will use lazy Dask arrays instead of in-memory numpy arrays

    Returns
    ----------
    ds_compound : :class:`numpy.ndarray` or :class:`dask.array.core.Array`
        2D complex array arranged as [sample, features]

    Examples
    --------
    >>> import numpy as np
    >>> import sidpy
    >>> real_val = np.hstack([5 * np.random.rand(6),
    >>>                       7 * np.random.rand(6)])
    >>> print('Real valued dataset of shape {}:'.format(real_val.shape))
    >>> print(real_val)
    Real valued dataset of shape (12,):
    [3.59249723 1.05674621 4.41035214 1.84720102 1.79672691 4.7636207
     3.09574246 0.76396171 3.38140637 4.97629028 0.83303717 0.32816285]

    >>> comp_val = sidpy.dtype_utils.stack_real_to_complex(real_val)
    >>> print('Complex-valued array of shape: {}'.format(comp_val.shape))
    >>> print(comp_val)
    Complex-valued array of shape: (6,)
    [3.59249723+3.09574246j 1.05674621+0.76396171j 4.41035214+3.38140637j
     1.84720102+4.97629028j 1.79672691+0.83303717j 4.7636207 +0.32816285j]
    """
    if not isinstance(ds_real, (np.ndarray, da.core.Array, h5py.Dataset)):
        if not isinstance(ds_real, (tuple, list)):
            raise TypeError("Expected at least an iterable like a list or tuple")
        ds_real = np.array(ds_real)
    if len(ds_real.dtype) > 0:
        raise TypeError("Array cannot have a compound dtype")
    if is_complex_dtype(ds_real.dtype):
        raise TypeError("Array cannot have complex dtype")

    if ds_real.shape[-1] / 2 != ds_real.shape[-1] // 2:
        raise ValueError("Last dimension must be even sized")
    half_point = ds_real.shape[-1] // 2

    if isinstance(ds_real, da.core.Array):
        lazy = True

    if lazy and not isinstance(ds_real, da.core.Array):
        ds_real = lazy_load_array(ds_real)

    return ds_real[..., :half_point] + 1j * ds_real[..., half_point:]


def stack_real_to_compound(ds_real, compound_type, lazy=False):
    """
    Converts a real-valued dataset to a compound dataset (along the last axis) of the provided compound d-type

    Parameters
    ------------
    ds_real : :class:`numpy.ndarray`, :class:`dask.array.core.Array`, or :class:`h5py.Dataset`
        n dimensional real-valued numpy array or HDF5 dataset where data arranged as [instance, features]
    compound_type : :class:`numpy.dtype`
        Target complex data-type
    lazy : bool, optional. Default = False
        If set to True, will use lazy Dask arrays instead of in-memory numpy arrays

    Returns
    ----------
    ds_compound : :class:`numpy.ndarray` or :class:`dask.array.core.Array`
        N-dimensional complex-valued array arranged as [sample, features]

    Examples
    --------
    >>> import numpy as np
    >>> import sidpy
    >>> struct_dtype = np.dtype({'names': ['r', 'g', 'b'],
    >>>                          'formats': [np.float32, np.uint16, np.float64]})
    >>> num_elems = 5
    >>> real_val = np.concatenate((np.random.random(size=num_elems) * 1024,
    >>>                            np.random.randint(0, high=1024, size=num_elems),
    >>>                            np.random.random(size=num_elems) * 1024))
    >>> print('Real valued dataset of shape {}:'.format(real_val.shape))
    >>> print(real_val)
    Real valued dataset of shape (15,):
    [276.65339095 527.80665658 741.38145798 647.06743252 710.41729083
     380.         796.         504.         355.         985.
     960.70015068 567.47024098 881.25140299 105.48936013 933.13686734]

    >>> comp_val = sidpy.dtype_utils.stack_real_to_compound(real_val, struct_dtype)

    >>> print('Structured array of shape: {}'.format(comp_val.shape))
    >>> print(comp_val)
    Structured array of shape: (5,)
    [(276.65338, 380, 960.70015068) (527.80664, 796, 567.47024098)
     (741.3815 , 504, 881.25140299) (647.06744, 355, 105.48936013)
     (710.4173 , 985, 933.13686734)]
    """
    if lazy or isinstance(ds_real, da.core.Array):
        raise NotImplementedError('Lazy operation not available due to absence of Dask support')
    if not isinstance(ds_real, (np.ndarray, h5py.Dataset)):
        if not isinstance(ds_real, (list, tuple)):
            raise TypeError("Expected at least an iterable like a list or tuple")
        ds_real = np.array(ds_real)
    if len(ds_real.dtype) > 0:
        raise TypeError("Array cannot have a compound dtype")
    elif is_complex_dtype(ds_real.dtype):
        raise TypeError("Array cannot have complex dtype")
    if not isinstance(compound_type, np.dtype):
        raise TypeError('Provided object must be a structured array dtype')

    new_spec_length = ds_real.shape[-1] / len(compound_type)
    if new_spec_length % 1:
        raise ValueError('Provided compound type was not compatible by number of elements')

    new_spec_length = int(new_spec_length)
    new_shape = list(ds_real.shape)  # Make mutable
    new_shape[-1] = new_spec_length

    xp = np
    kwargs = {}
    """
    if isinstance(ds_real, h5py.Dataset) and not lazy:
        warn('HDF5 datasets will be loaded as Dask arrays in the future. ie - kwarg lazy will default to True in future releases of sidpy')
    if isinstance(ds_real, da.core.Array):
        lazy = True    
    if lazy:
        xp = da
        ds_real = lazy_load_array(ds_real)
        kwargs = {'chunks': 'auto'}
    """

    ds_compound = xp.empty(new_shape, dtype=compound_type, **kwargs)
    for name_ind, name in enumerate(compound_type.names):
        i_start = name_ind * new_spec_length
        i_end = (name_ind + 1) * new_spec_length
        ds_compound[name] = ds_real[..., i_start:i_end]

    return ds_compound.squeeze()


def stack_real_to_target_dtype(ds_real, new_dtype, lazy=False):
    """
    Transforms real data into the target dtype

    Parameters
    ----------
    ds_real : :class:`numpy.ndarray`, :class:`dask.array.core.Array` or :class:`h5py.Dataset`
        n dimensional real-valued numpy array or HDF5 dataset
    new_dtype : :class:`numpy.dtype`
        Target data-type

    Returns
    ----------
    ret_val :  :class:`numpy.ndarray` or :class:`dask.array.core.Array`
        N-dimensional array of the target data-type

    Examples
    --------
    >>> import numpy as np
    >>> import sidpy
    >>> struct_dtype = np.dtype({'names': ['r', 'g', 'b'],
    >>>                          'formats': [np.float32, np.uint16, np.float64]})
    >>> num_elems = 5
    >>> real_val = np.concatenate((np.random.random(size=num_elems) * 1024,
    >>>                            np.random.randint(0, high=1024, size=num_elems),
    >>>                            np.random.random(size=num_elems) * 1024))
    >>> print('Real valued dataset of shape {}:'.format(real_val.shape))
    >>> print(real_val)
    Real valued dataset of shape (15,):
    [276.65339095 527.80665658 741.38145798 647.06743252 710.41729083
     380.         796.         504.         355.         985.
     960.70015068 567.47024098 881.25140299 105.48936013 933.13686734]

    >>> comp_val = sidpy.dtype_utils.stack_real_to_target_dtype(real_val, struct_dtype)

    >>> print('Structured array of shape: {}'.format(comp_val.shape))
    >>> print(comp_val)
    Structured array of shape: (5,)
    [(276.65338, 380, 960.70015068) (527.80664, 796, 567.47024098)
     (741.3815 , 504, 881.25140299) (647.06744, 355, 105.48936013)
     (710.4173 , 985, 933.13686734)]
    """
    if is_complex_dtype(new_dtype):
        return stack_real_to_complex(ds_real, lazy=lazy)
    try:
        if len(new_dtype) > 0:
            return stack_real_to_compound(ds_real, new_dtype, lazy=lazy)
    except TypeError:
        return new_dtype(ds_real)

    # catching all other cases, such as np.dtype('<f4')
    return new_dtype.type(ds_real)


def validate_dtype(dtype):
    """
    Checks the provided object to ensure that it is a valid dtype that can be written to an HDF5 file.
    Raises a type error if invalid. Returns True if the object passed the tests

    Parameters
    ----------
    dtype : object
        Object that is hopefully a :class:`h5py.Datatype`, or :class:`numpy.dtype` object

    Returns
    -------
    status : bool
        True if the object was a valid data-type

    Examples
    --------
    >>> import numpy as np
    >>> import sidpy
    >>> for item in [np.float16, np.complex64, np.uint8, np.int16]:
    >>>     print('Is {} a valid dtype? : {}'.format(item, sidpy.dtype_utils.validate_dtype(item)))
    Is <class 'numpy.float16'> a valid dtype? : True
    Is <class 'numpy.complex64'> a valid dtype? : True
    Is <class 'numpy.uint8'> a valid dtype? : True
    Is <class 'numpy.int16'> a valid dtype? : True

    # This function is especially useful on compound or structured data types:
    >>> struct_dtype = np.dtype({'names': ['r', 'g', 'b'],
    >>>                         'formats': [np.float32, np.uint16, np.float64]})
    >>> print('Is {} a valid dtype? : {}'.format(struct_dtype, sidpy.dtype_utils.validate_dtype(struct_dtype)))
    Is [('r', '<f4'), ('g', '<u2'), ('b', '<f8')] a valid dtype? : True
    """
    if isinstance(dtype, (h5py.Datatype, np.dtype)):
        pass
    elif isinstance(np.dtype(dtype), np.dtype):
        # This should catch all those instances when dtype is something familiar like - np.float32
        pass
    else:
        raise TypeError('dtype should either be a numpy or h5py dtype')
    return True


def is_complex_dtype(dtype):
    """
    Checks if the provided dtype is a complex dtype

    Parameters
    ----------
    dtype : object
        Object that is a class:`h5py.Datatype`, or :class:`numpy.dtype` object

    Returns
    -------
    is_complex : bool
        True if the dtype was a complex dtype. Else returns False

    Examples
    --------
    >>> import numpy as np
    >>> import sidpy
    >>> for dtype in [np.float32, np.float16, np.uint8, np.int16, bool]:
    >>>     print('Is {} a complex dtype?: {}'.format(dtype, (sidpy.dtype_utils.is_complex_dtype(dtype))))
    Is <class 'numpy.float32'> a complex dtype?: False
    Is <class 'numpy.float16'> a complex dtype?: False
    Is <class 'numpy.uint8'> a complex dtype?: False
    Is <class 'numpy.int16'> a complex dtype?: False
    Is <class 'bool'> a complex dtype?: False

    >>> struct_dtype = np.dtype({'names': ['r', 'g', 'b'],
    >>>                         'formats': [np.float32, np.uint16, np.float64]})
    Is [('r', '<f4'), ('g', '<u2'), ('b', '<f8')] a complex dtype?: False

    >>> for dtype in [complex, np.complex64, np.complex128, np.complex256]:
    >>>     print('Is {} a complex dtype?: {}'.format(dtype, (sidpy.dtype_utils.is_complex_dtype(dtype))))
    Is <class 'complex'> a complex dtype?: True
    Is <class 'numpy.complex64'> a complex dtype?: True
    Is <class 'numpy.complex128'> a complex dtype?: True
    Is <class 'numpy.complex256'> a complex dtype?: False
    """
    validate_dtype(dtype)
    if dtype in [complex, np.complex64, np.complex128]:
        return True
    return False
