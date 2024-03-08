# -*- coding: utf-8 -*-
"""
Simple yet handy HDF5 utilities, independent of the  data model

Created on Tue Nov  3 21:14:25 2015

@author: Suhas Somnath, Chris Smith
"""
from __future__ import division, print_function, absolute_import, unicode_literals
import socket
import sys
from warnings import warn
from platform import platform
from enum import Enum
import h5py
import numpy as np
from dask import array as da

from sidpy.__version__ import version as sidpy_version
from sidpy.base.string_utils import validate_single_string_arg, \
    validate_list_of_strings, clean_string_att, get_time_stamp
# from sidpy.base.dict_utils import flatten_dict

if sys.version_info.major == 3:
    unicode = str


def print_tree(parent, rel_paths=False):
    """
    Simple function to recursively print the contents of a hdf5 group

    Parameters
    ----------
    parent : :class:`h5py.Group`
        HDF5 (sub-)tree to print
    rel_paths : bool, optional. Default = False
        True - prints the relative paths for all elements.
        False - prints a tree-like structure with only the element names
    """
    # TODO: Accept callables where the user could filter out group / datasets
    # based on some condition. This will simplify print_tree extensions in
    # pyUSID and pyNSID
    if not isinstance(parent, (h5py.File, h5py.Group)):
        raise TypeError('Provided object is not a h5py.File or h5py.Group '
                        'object')

    def __print(name, obj):
        if rel_paths:
            print(name)
        else:
            levels = name.count('/')
            curr_name = name[name.rfind('/') + 1:]

            print(levels * '  ' + '├ ' + curr_name)
            if isinstance(obj, h5py.Group):
                print((levels + 1) * '  ' + len(curr_name) * '-')

    print(parent.name)
    parent.visititems(__print)


def get_auxiliary_datasets(h5_object, aux_dset_name=None):
    """
    Returns auxiliary dataset objects associated with some DataSet through its attributes.
    Note - region references will be ignored.

    Parameters
    ----------
    h5_object : :class:`h5py.Dataset`, :class:`h5py.Group` or :class:`h5py.File`
        Dataset object reference.
    aux_dset_name : str or :class:`list` of str, optional. Default = all
        Name of auxiliary :class:`h5py.Dataset` objects to return.

    Returns
    -------
    list of :class:`h5py.Reference` of auxiliary :class:`h5py.Dataset` objects.

    """
    if not isinstance(h5_object, (h5py.Dataset, h5py.Group, h5py.File)):
        raise TypeError('h5_object should be a h5py.Dataset, h5py.Group or h5py.File object')

    if aux_dset_name is None:
        aux_dset_name = h5_object.attrs.keys()
    else:
        aux_dset_name = validate_list_of_strings(aux_dset_name, 'aux_dset_name')

    data_list = list()
    curr_name = None
    try:
        h5_file = h5_object.file
        for curr_name in aux_dset_name:
            h5_ref = h5_object.attrs[curr_name]
            if isinstance(h5_ref, h5py.Reference) and isinstance(h5_file[h5_ref], h5py.Dataset) and not \
                    isinstance(h5_ref, h5py.RegionReference):
                data_list.append(h5_file[h5_ref])
    except KeyError:
        raise KeyError('%s is not an attribute of %s' % (str(curr_name), h5_object.name))

    return data_list


def get_attr(h5_object, attr_name):
    """
    Returns the attribute from the h5py object

    Parameters
    ----------
    h5_object : :class:`h5py.Dataset`, :class:`h5py.Group` or :class:`h5py.File`
        object whose attribute is desired
    attr_name : str
        Name of the attribute of interest

    Returns
    -------
    att_val : object
        value of attribute, in certain cases (byte strings or list of byte strings) reformatted to readily usable forms

    """
    if not isinstance(h5_object, (h5py.Dataset, h5py.Group, h5py.File)):
        raise TypeError('h5_object should be a h5py.Dataset, h5py.Group or h5py.File object')

    attr_name = validate_single_string_arg(attr_name, 'attr_name')

    if attr_name not in h5_object.attrs.keys():
        raise KeyError("'{}' is not an attribute in '{}'".format(attr_name, h5_object.name))

    h5py_major = int(h5py.__version__.split('.')[0])

    att_val = h5_object.attrs.get(attr_name)
    if isinstance(att_val, np.bytes_) or isinstance(att_val, bytes):
        att_val = att_val.decode('utf-8')

    elif isinstance(att_val, np.ndarray):
        if sys.version_info.major == 3:
            if att_val.dtype.type in [np.bytes_]:
                att_val = np.array([str(x, 'utf-8') for x in att_val])
            elif att_val.dtype.type in [np.object_] and h5py_major < 3:
                att_val = np.array([str(x, 'utf-8') for x in att_val])

    return att_val


def get_attributes(h5_object, attr_names=None, strict=False):
    """
    Returns attribute associated with some DataSet.

    Parameters
    ----------
    h5_object : :class:`h5py.Dataset`
        Dataset object reference.
    attr_names : str or :class:`list` of str, optional. Default = all
        Name of attribute object to return.
    strict : bool, optional. Default = False
        If True - raises a KeyError if desired keys are not found.
        Else, raises warning instead.
        This is especially useful when attempting to read attributes with
        invalid names such as spaces on either sides of text.

    Returns
    -------
    att_dict : dict
        Dictionary containing (name,value) pairs of attributes

    """
    if not isinstance(h5_object, (h5py.Dataset, h5py.Group, h5py.File)):
        raise TypeError('h5_object should be a h5py.Dataset, h5py.Group or h5py.File object')

    if attr_names is None:
        attr_names = h5_object.attrs.keys()
    else:
        attr_names = validate_list_of_strings(attr_names, 'attr_names')
        # Set strict to True since user is looking for specific attributes
        strict = True

    att_dict = {}

    for attr in attr_names:
        try:
            att_dict[attr] = get_attr(h5_object, attr)
        except KeyError:
            message = '"{}" is not an attribute of {}'.format(attr, h5_object.name)
            if strict:
                raise KeyError(message)
            else:
                warn(message)

    return att_dict


def get_h5_obj_refs(obj_names, h5_refs):
    """
    Given a list of H5 references and a list of names,
    this method returns H5 objects corresponding to the names

    Parameters
    ----------
    obj_names : string or List of strings
        names of target h5py objects
    h5_refs : H5 object reference or List of H5 object references
        list containing the target reference

    Returns
    -------
    found_objects : List of HDF5 dataset references
        Corresponding references

    """
    obj_names = validate_list_of_strings(obj_names, 'attr_names')

    if isinstance(h5_refs, (h5py.File, h5py.Group, h5py.Dataset)):
        h5_refs = [h5_refs]
    if not isinstance(h5_refs, (list, tuple)):
        raise TypeError('h5_refs should be a / list of h5py.Dataset, h5py.Group or h5py.File object(s)')

    found_objects = []
    for target_name in obj_names:
        for h5_object in h5_refs:
            if not isinstance(h5_object, (h5py.File, h5py.Group, h5py.Dataset)):
                continue
            if h5_object.name.split('/')[-1] == target_name:
                found_objects.append(h5_object)

    return found_objects


def validate_h5_objs_in_same_h5_file(h5_src, h5_other):
    """
    Checks if the provided objects are in the same HDF5 file.
    If not, it throws a ValueError

    Parameters
    ----------
    h5_src : h5py.Dataset, h5py.File, or h5py.Group object
        First object to compare
    h5_other : h5py.Dataset, h5py.File, or h5py.Group object
        Second object to compare
    """
    if not isinstance(h5_src, (h5py.Dataset, h5py.File, h5py.Group)):
        raise TypeError('h5_src should either be a h5py Dataset, File, or '
                        'Group')
    if not isinstance(h5_other, (h5py.Dataset, h5py.File, h5py.Group)):
        raise TypeError('h5_other should either be a h5py Dataset, File, or'
                        ' Group')
    if h5_src.file != h5_other.file:
        raise ValueError('Cannot link h5 objects across files. '
                         '{} is present in file: {}, while {} is in file :'
                         '{}'.format(h5_src.name, h5_src.file, h5_other.name,
                                     h5_other.file))


def __link_h5_obj(h5_src, h5_other, alias=None):
    validate_h5_objs_in_same_h5_file(h5_src, h5_other)
    if alias is None:
        alias = h5_other.name.split('/')[-1]
    h5_src.attrs[alias] = h5_other.ref


def link_h5_objects_as_attrs(src, h5_objects):
    """
    Creates Dataset attributes that contain references to other Dataset Objects.

    Parameters
    -----------
    src : Reference to h5.object
        Reference to the object to which attributes will be added
    h5_objects : list of references to h5.objects
        objects whose references that can be accessed from src.attrs

    Returns
    --------
    None

    """
    if not isinstance(src, (h5py.Dataset, h5py.File, h5py.Group)):
        raise TypeError('src should either be a h5py Dataset, File, or Group')
    if isinstance(h5_objects, (h5py.Dataset, h5py.Group)):
        h5_objects = [h5_objects]

    for itm in h5_objects:
        if not isinstance(itm, (h5py.Dataset, h5py.Group)):
            raise TypeError('h5_objects should only contain h5py. Dataset and Group objects')
        __link_h5_obj(src, itm)


def link_h5_obj_as_alias(h5_main, h5_ancillary, alias_name):
    """
    Creates Dataset attributes that contain references to other Dataset Objects.
    This function is useful when the reference attribute must have a reserved name.
    Such as linking 'SHO_Indices' as 'Spectroscopic_Indices'

    Parameters
    ------------
    h5_main : h5py.Dataset
        Reference to the object to which attributes will be added
    h5_ancillary : h5py.Dataset
        object whose reference that can be accessed from src.attrs
    alias_name : String
        Alias / alternate name for trg

    """
    if not isinstance(h5_main, (h5py.Dataset, h5py.File, h5py.Group)):
        raise TypeError('h5_main should either be a h5py Dataset, File, or Group')
    if not isinstance(h5_ancillary, (h5py.Dataset, h5py.Group)):
        raise TypeError('h5_ancillary should be a h5py. Dataset or Group object')
    alias_name = validate_single_string_arg(alias_name, 'alias_name')

    __link_h5_obj(h5_main, h5_ancillary, alias=alias_name)


def is_editable_h5(h5_obj):
    """
    Returns True if the file containing the provided h5 object is in w or r+ modes

    Parameters
    ----------
    h5_obj : h5py.File, h5py.Group, or h5py.Dataset object
        h5py object

    Returns
    -------
    mode : bool
        True if the file containing the provided h5 object is in w or r+ modes

    """
    if not isinstance(h5_obj, (h5py.File, h5py.Group, h5py.Dataset)):
        raise TypeError('h5_obj should be a h5py File, Group or Dataset object but is instead of type '
                        '{}t'.format(type(h5_obj)))
    try:
        file_handle = h5_obj.file
    except RuntimeError:
        raise ValueError('Encountered a RuntimeError possibly due to a closed file')
    # file handle is actually an open hdf file

    if file_handle.mode == 'r':
        return False
    return True


def write_book_keeping_attrs(h5_obj):
    """
    Writes basic bookkeeping and posterity related attributes to groups
    created using sidpy such as machine id, version, timestamp.

    Parameters
    ----------
    h5_obj : :class:`h5py.Dataset`, :class:`h5py.Group`, or :class:`h5py.File`
        Object to which basic bookkeeping attributes need to be written

    """
    if not isinstance(h5_obj, (h5py.Group, h5py.File, h5py.Dataset)):
        raise TypeError('h5_obj should be a h5py.Group, h5py.File, or h5py.Dataset object')
    write_simple_attrs(h5_obj, {'machine_id': socket.getfqdn(),
                                'timestamp': get_time_stamp(),
                                'platform': platform(),
                                'sidpy_version': sidpy_version},
                       verbose=False)


def write_simple_attrs(h5_obj, attrs, force_to_str=True, verbose=False):
    """
    Writes attributes to a h5py object

    Parameters
    ----------
    h5_obj : :class:`h5py.File`, :class:`h5py.Group`, or h5py.Dataset object
        h5py object to which the attributes will be written to
    attrs : dict
        Dictionary containing the attributes as key-value pairs
    force_to_str : bool, optional. Default = True
        Whether or not to cast keys or values to string when they do not have
        the correct types
    verbose : bool, optional. Default=False
        Whether or not to print debugging statements

    """
    if not isinstance(attrs, dict):
        raise TypeError('attrs should be a dictionary but is instead of type '
                        '{}'.format(type(attrs)))
    if not isinstance(h5_obj, (h5py.File, h5py.Group, h5py.Dataset)):
        raise TypeError('h5_obj should be a h5py File, Group or Dataset object'
                        ' but is instead of type '
                        '{}t'.format(type(h5_obj)))

    for key, val in attrs.items():
        if not isinstance(key, (str, unicode)):
            if force_to_str:
                warn('Converted key: {} from type: {} to str'
                     ''.format(key, type(key)))
                key = str(key)
            else:
                warn('Skipping attribute with key: {}. Expected str, got {}'
                     ''.format(key, type(key)))
                continue

        # Get rid of spaces in the key
        key = key.strip()

        if val is None:
            continue
        if isinstance(val, Enum):
            if verbose:
                print('taking the name: {} of Enum: {}'.format(val.name, val))
            val = val.name

        if isinstance(val, list):
            dictionaries = False
            for item in val:
                if isinstance(item, dict):
                    dictionaries = True
                    break
            if dictionaries:
                new_val = {}
                for key2, item in enumerate(val):
                    new_val[str(key2)] = item
                val = new_val

        if isinstance(val, dict):
            if isinstance(h5_obj, h5py.Dataset):
                raise ValueError('provided dictionary was nested, not flat. '
                                 'Flatten dictionary using sidpy.base.dict_utils.'
                                 'flatten_dict before calling sidpy.hdf.hdf_utils.'
                                 'write_simple_attrs')
            else:
                new_object = h5_obj.create_group(str(key))
                write_simple_attrs(new_object, val, force_to_str=True, verbose=False)
            
        if verbose:
            print('Writing attribute: {} with value: {}'.format(key, val))
        
        if not (isinstance(val, dict)):  # not sure how this can happen
            if verbose:
                print(key, val)
            clean_val = clean_string_att(val)
            
            if verbose:
                print('Attribute cleaned into: {}'.format(clean_val))
            try:
                h5_obj.attrs[key] = clean_val
            except Exception as excp:
                if verbose:
                    if force_to_str:
                        warn('Casting attribute value: {} of type: {} to str'.format(val, type(val)))
                        h5_obj.attrs[key] = str(val)
                    else:
                        raise excp('Could not write attribute value: {} of type: {}'.format(val, type(val)))

    if verbose:
        print('Wrote all (simple) attributes to {}: {}\n'
              ''.format(type(h5_obj), h5_obj.name.split('/')[-1]))


def lazy_load_array(dataset):
    """
    Loads the provided object as a dask array (h5py.Dataset or numpy.ndarray)

    Parameters
    ----------
    dataset : :class:`numpy.ndarray`, or :class:`h5py.Dataset`, or
        :class:`dask.array.core.Array` to load as dask array

    Returns
    -------
    :class:`dask.array.core.Array`
        Dask array with appropriate chunks
    """
    if isinstance(dataset, da.core.Array):
        return dataset
    elif not isinstance(dataset, (h5py.Dataset, np.ndarray)):
        raise TypeError('Expected one of h5py.Dataset, dask.array.core.Array, or numpy.ndarray'
                        'objects. Provided object was of type: {}'.format(type(dataset)))
    # Cannot pass 'auto' for chunks for python 2!
    chunks = "auto" if sys.version_info.major == 3 else dataset.shape
    if isinstance(dataset, h5py.Dataset):
        chunks = chunks if dataset.chunks is None else dataset.chunks
    return da.from_array(dataset, chunks=chunks)


def copy_attributes(source, dest, skip_refs=True, verbose=False):
    """
    Copy attributes from one h5object to another

    Parameters
    ----------
    source : h5py.Dataset, :class:`h5py.Group`, or :class:`h5py.File`
        Object containing the desired attributes
    dest : h5py.Dataset, :class:`h5py.Group`, or :class:`h5py.File`
        Object to which the attributes need to be copied to
    skip_refs : bool, optional. default = True
        Whether or not the references (dataset and region) should be skipped
    verbose : bool, optional. Default = False
        Whether or not to print logs for debugging
    """
    message = 'should be a h5py.Dataset, h5py.Group,or h5py.File object'
    if not isinstance(source, (h5py.Dataset, h5py.Group, h5py.File)):
        raise TypeError('source ' + message)
    if not isinstance(dest, (h5py.Dataset, h5py.Group, h5py.File)):
        raise TypeError('dest ' + message)

    skip_dset_refs = skip_refs
    try:
        validate_h5_objs_in_same_h5_file(source, dest)
    except ValueError:
        if not skip_refs:
            warn('Dataset references will not be copied since {} and {} are '
                 'in different files'.format(source, dest))
        skip_dset_refs = True

    for att_name in source.attrs.keys():
        # print(att_name)
        if att_name not in ['DIMENSION_LIST']:
            att_val = get_attr(source, att_name)
            """
            Don't copy references unless asked
            """
            if isinstance(att_val, h5py.Reference) and not isinstance(att_val, h5py.RegionReference):
                if not skip_dset_refs:
                    if verbose:
                        print('dset ref copying ' + att_name)
                    dest.attrs[att_name] = att_val
            elif isinstance(att_val, h5py.RegionReference):
                # handled in dedicated if condition below
                continue
            else:
                # everything else
                if verbose:
                    print('simple copying ' + att_name)
                dest.attrs[att_name] = clean_string_att(att_val)

    if not skip_refs:
        # This can be copied across files without problems
        mesg = 'Could not copy region references to {}.'.format(dest.name)
        if isinstance(dest, h5py.Dataset):
            try:
                if verbose:
                    print('requested reg ref copy')
                # copy_region_refs(source, dest)
                pass  # TODO: activate again

            except TypeError:
                warn(mesg)
        else:
            warn('Cannot copy region references to {}'.format(type(dest)))

    return dest


def copy_dataset(h5_orig_dset, h5_dest_grp, alias=None, verbose=False):
    """
    Copies the provided HDF5 dataset to the provided destination. This function
    is handy when needing to make copies of datasets to a different HDF5 file.
    Notes
    -----
    This function does NOT copy all linked objects such as ancillary
    datasets. Call `copy_linked_objects` to accomplish that goal.
    Parameters
    ----------
    h5_orig_dset : h5py.Dataset
    h5_dest_grp : h5py.Group or h5py.File object :
        Destination where the duplicate dataset will be created
    alias : str, optional. Default = name from `h5_orig_dset`:
        Name to be assigned to the copied dataset
    verbose : bool, optional. Default = False
        Whether or not to print logs to assist in debugging
    Returns
    -------
    """
    if not isinstance(h5_orig_dset, h5py.Dataset):
        raise TypeError("'h5_orig_dset' should be a h5py.Dataset object")
    if not isinstance(h5_dest_grp, (h5py.File, h5py.Group)):
        raise TypeError("'h5_dest_grp' should either be a h5py.File or "
                        "h5py.Group object")
    if alias is not None:
        validate_single_string_arg(alias, 'alias')
    else:
        alias = h5_orig_dset.name.split('/')[-1]

    if alias in h5_dest_grp.keys():
        if verbose:
            warn('{} already contains an object with the same name: {}'
                 ''.format(h5_dest_grp, alias))
        h5_new_dset = h5_dest_grp[alias]
        if not isinstance(h5_new_dset, h5py.Dataset):
            raise TypeError('{} already contains an object: {} with the desired'
                            ' name which is not a dataset'.format(h5_dest_grp,
                                                                  h5_new_dset))

        da_source = lazy_load_array(h5_orig_dset)
        da_dest = lazy_load_array(h5_new_dset)

        if da_source.shape != da_dest.shape:
            raise ValueError('Existing dataset: {} has a different shape '
                             'compared to the original dataset: {}'
                             ''.format(h5_new_dset, h5_orig_dset))
        if not da.allclose(da_source, da_dest):
            raise ValueError('Existing dataset: {} has different contents'
                             'compared to the original dataset: {}'
                             ''.format(h5_new_dset, h5_orig_dset))
    else:

        kwargs = {'shape': h5_orig_dset.shape,
                  'dtype': h5_orig_dset.dtype,
                  'compression': h5_orig_dset.compression,
                  'chunks': h5_orig_dset.chunks}
        if h5_orig_dset.file.driver == 'mpio':
            if kwargs.pop('compression', None) is not None:
                warn('This HDF5 file has been opened wth the '
                     '"mpio" communicator. mpi4py does not allow '
                     'creation of compressed datasets. Compression'
                     ' kwarg has been removed')
        if verbose:
            print('Creating new HDF5 dataset named: {} at: {} with'
                  ' kwargs: {}'.format(alias, h5_dest_grp,
                                       kwargs))
        h5_new_dset = h5_dest_grp.create_dataset(alias,
                                                 **kwargs)
        if verbose:
            print('dask.array will copy data from source dataset '
                  'to new dataset')
        da.to_hdf5(h5_new_dset.file.filename,
                   {h5_new_dset.name: lazy_load_array(h5_orig_dset)})
    if verbose:
        print('Copying simple attributes of original dataset: {} to '
              'destination dataset: {}'.format(h5_orig_dset, h5_new_dset))

    copy_attributes(h5_orig_dset, h5_new_dset, skip_refs=True)
    # TODO: reinstate copy all region_refs()
    # copy_all_region_refs(h5_orig_dset, h5_new_dset)

    return h5_new_dset


def copy_linked_objects(h5_source, h5_dest, verbose=False):
    """
    Recursively copies datasets linked to the source h5 object to the
    destination h5 object that are in different HDF5 files.

    This is for copying ancillary datasets to a target dataset that is
    missing ancillary datasets. It is not meant for copying to a Group,
    but that is supported.
    Notes
    -----
    We anticipate this function being used to copy over ancillary datasets
    Parameters
    ----------
    h5_source : h5py.Dataset or h5py.Group object
        Source object
    h5_dest : h5py.Dataset or h5py.Group object
        Destination object
    verbose : bool, optional. Default: False
        Whether or not to print logs for debugging purposes
    """
    try:
        # The following line takes care of object validation
        validate_h5_objs_in_same_h5_file(h5_source, h5_dest)
        same_file = True
    except ValueError:
        same_file = False

    if same_file:
        warn('{} and {} are in the same HDF5 file. Consider copying references'
             ' instead of copying linked objects'.format(h5_source, h5_dest))
        return

    if isinstance(h5_dest, h5py.Group):
        h5_dest_grp = h5_dest
    else:
        h5_dest_grp = h5_dest.parent

    # Now we are working on other files
    for link_obj_name in h5_source.attrs.keys():
        h5_orig_obj = get_attr(h5_source, link_obj_name)
        if isinstance(h5_orig_obj, h5py.Reference) and not \
                isinstance(h5_orig_obj, h5py.RegionReference):
            h5_orig_obj = h5_source.file[h5_orig_obj]
            if verbose:
                print('Attempting to copy object linked to source: {} as {}'
                      ''.format(h5_orig_obj, link_obj_name))
            # Check to see if such a dataset already exist
            if link_obj_name in h5_dest_grp.keys():
                h5_new_obj = h5_dest_grp[link_obj_name]
                warn('An object with the same name: {} already exists in the '
                     'destination group: {}'.format(h5_new_obj, h5_dest_grp.name))
                if type(h5_dest_grp[link_obj_name]) != type(h5_orig_obj):
                    mesg = 'Destination parent: {} already has a child named' \
                           ' {} that is of type: {} which does not match ' \
                           'with that of the object linked with the source ' \
                           'dataset: {}'.format(h5_dest_grp, link_obj_name,
                                                type(h5_orig_obj),
                                                type(h5_new_obj))
                    raise TypeError(mesg)

                elif isinstance(h5_new_obj, h5py.Dataset):
                    _ = copy_dataset(h5_orig_obj, h5_dest_grp,
                                     alias=link_obj_name, verbose=verbose)
                    h5_dest.attrs[link_obj_name] = h5_new_obj.ref
                    continue
                elif isinstance(h5_new_obj, h5py.Group):
                    raise ValueError('Destination already contains another '
                                     'HDF5 group: {} with the same name as '
                                     'the source: {}'.format(h5_new_obj,
                                                             h5_orig_obj))
                else:
                    raise NotImplementedError('Unable to copy {} objects yet'
                                              '. Contact developer if you need'
                                              ' this'
                                              ''.format(type(h5_orig_obj)))
            else:
                if isinstance(h5_orig_obj, h5py.Dataset):
                    h5_new_obj = copy_dataset(h5_orig_obj, h5_dest_grp,
                                              alias=link_obj_name,
                                              verbose=verbose)
                    h5_dest.attrs[link_obj_name] = h5_new_obj.ref
                else:
                    raise NotImplementedError('Unable to copy {} objects yet'
                                              '. Contact developer if you need'
                                              ' this'.format(type(h5_orig_obj)))


def find_dataset(h5_group, dset_name):
    """
    Uses visit() to find all datasets with the desired name

    Parameters
    ----------
    h5_group : :class:`h5py.Group`
        Group to search within for the Dataset
    dset_name : str
        Name of the dataset to search for

    Returns
    -------
    datasets : list
        List of [Name, object] pairs corresponding to datasets that match `ds_name`.

    """
    if not isinstance(h5_group, (h5py.File, h5py.Group)):
        raise TypeError('h5_group should be a h5py.File or h5py.Group object')
    dset_name = validate_single_string_arg(dset_name, 'dset_name')

    # print 'Finding all instances of', ds_name
    datasets = []

    def __find_name(name, obj):
        if dset_name in name.split('/')[-1] and isinstance(obj, h5py.Dataset):
            datasets.append(obj)
        return

    h5_group.visititems(__find_name)

    return datasets


def write_dict_to_h5_group(h5_group, metadata, group_name):
    """
    If the provided metadata parameter is a non-empty dictionary, this function
    will create a HDF5 group called group_name within the provided h5_group and
    write the contents of metadata into the newly created group
    Parameters
    ----------
    h5_group : h5py.Group
        Parent group to write metadata into
    metadata : dict
        Dictionary that needs to be written into the group
    group_name : str
        Name of the group to write attributes into


    Returns
    -------
    h5_metadata_grp : h5py.Group
        Handle to the newly create group containing the metadata

    Notes
    -----
    Writes now (sidpy version 0.0.6) nested dictionaries to HDF5 files.
    Use h5_group_to_dict to read from HDF5 file.
    """
    if not isinstance(metadata, dict):
        raise TypeError('metadata is not a dict but of type: {}'
                        ''.format(type(metadata)))
    if len(metadata) < 1:
        return None
    if not isinstance(h5_group, (h5py.Group, h5py.File)):
        raise TypeError('h5_group is neither a h5py.Group or h5py.File object'
                        'and is of type: {}'.format(type(h5_group)))

    validate_single_string_arg(group_name, 'group_name')
    group_name = group_name.replace(' ', '_')
    h5_md_group = h5_group.create_group(group_name)
    # flat_dict = flatten_dict(metadata)
    write_simple_attrs(h5_md_group, metadata)
    return h5_md_group


def h5_group_to_dict(group_iter, group_dict={}):
    """ 
    Reads a hdf5 group into a nested dictionary
    
    Parameters
    ----------
    group_iter: hdf5.Group
        starting group to read from
    group_dict: dict
        group dictionary; mostly needed for recursive reading of nested groups but can be used for initialization
    Returns
    -------
    group_dict: dict
    """

    if not isinstance(group_iter, h5py.Group):
        raise TypeError('we need a h5py group to read from. Type given was {}'.format(type(group_iter)))
    if not isinstance(group_dict, dict):
        raise TypeError('group_dict needs to be a python dictionary')
   
    group_dict[group_iter.name.split('/')[-1]] = dict(group_iter.attrs)
    
    for key in group_iter.keys():
        h5_group_to_dict(group_iter[key], group_dict[group_iter.name.split('/')[-1]])
    return group_dict
