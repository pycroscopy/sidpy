# -*- coding: utf-8 -*-
"""
Tools for tracking provenance within HDF5 files

Created on Tue Nov  3 21:14:25 2015

@author: Suhas Somnath, Chris Smith
"""
from __future__ import division, print_function, absolute_import, \
    unicode_literals
import sys
from warnings import warn
import h5py
import numpy as np

if sys.version_info.major == 3:
    from collections.abc import Iterable
    unicode = str
else:
    from collections import Iterable

from sidpy.base.string_utils import validate_single_string_arg
from sidpy.hdf.hdf_utils import get_attr, write_book_keeping_attrs, \
    write_simple_attrs


def assign_group_index(h5_parent_group, base_name, verbose=False):
    """
    Searches the parent h5 group to find the next available index for the group

    Parameters
    ----------
    h5_parent_group : :class:`h5py.Group` object
        Parent group under which the new group object will be created
    base_name : str or unicode
        Base name of the new group without index
    verbose : bool, optional. Default=False
        Whether or not to print debugging statements

    Returns
    -------
    base_name : str or unicode
        Base name of the new group with the next available index as a suffix
    """
    if not isinstance(h5_parent_group, h5py.Group):
        raise TypeError('h5_parent_group should be a h5py.Group object')
    base_name = validate_single_string_arg(base_name, 'base_name')

    if len(base_name) == 0:
        raise ValueError('base_name should not be an empty string')

    if not base_name.endswith('_'):
        base_name += '_'

    temp = [key for key in h5_parent_group.keys()]
    if verbose:
        print('Looking for group names starting with {} in parent containing items: '
              '{}'.format(base_name, temp))
    previous_indices = []
    for item_name in temp:
        if isinstance(h5_parent_group[item_name], h5py.Group) and item_name.startswith(base_name):
            previous_indices.append(int(item_name.replace(base_name, '')))
    previous_indices = np.sort(previous_indices)
    if verbose:
        print('indices of existing groups with the same prefix: {}'.format(previous_indices))
    if len(previous_indices) == 0:
        index = 0
    else:
        index = previous_indices[-1] + 1
    return base_name + '{:03d}'.format(index)


def create_indexed_group(h5_parent_group, base_name):
    """
    Creates a group with an indexed name (eg - 'Measurement_012') under
    ``h5_parent_group`` using the provided ``base_name`` as a prefix for the
    group's name

    Parameters
    ----------
    h5_parent_group : :class:`h5py.Group` or :class:`h5py.File`
        File or group within which the new group will be created
    base_name : str or unicode
        Prefix for the group name. This need not end with a '_'. It will be
        added automatically
    """
    if not isinstance(h5_parent_group, (h5py.Group, h5py.File)):
        raise TypeError('h5_parent_group should be a h5py.File or Group object')
    base_name = validate_single_string_arg(base_name, 'base_name')

    group_name = assign_group_index(h5_parent_group, base_name)
    h5_new_group = h5_parent_group.create_group(group_name)
    write_book_keeping_attrs(h5_new_group)
    return h5_new_group


def create_results_group(h5_main, tool_name, h5_parent_group=None):
    """
    Creates a h5py.Group object auto-indexed and named as
    'DatasetName-ToolName_00x'

    Parameters
    ----------
    h5_main : h5py.Dataset object
        Reference to the dataset based on which the process / analysis is being
        performed
    tool_name : string / unicode
        Name of the Process / Analysis applied to h5_main
    h5_parent_group : h5py.Group, optional. Default = None
        Parent group under which the results group will be created. Use this
        option to write results into a new HDF5 file. By default, results will
        be written into the same group containing `h5_main`

    Returns
    -------
    h5_group : :class:`h5py.Group`
        Results group which can now house the results datasets
    """
    # TODO: Revise significantly. Avoid parent dataset name
    # Consider embedding refs to source datasets as attributes of group

    warn('The behavior of create_results_group is very likely to change soon '
         'and significantly. Use this function with caution', FutureWarning)

    if not isinstance(h5_main, h5py.Dataset):
        raise TypeError('h5_main should be a h5py.Dataset object')
    if h5_parent_group is not None:
        if not isinstance(h5_parent_group, (h5py.File, h5py.Group)):
            raise TypeError("'h5_parent_group' should either be a h5py.File "
                            "or h5py.Group object")
    else:
        h5_parent_group = h5_main.parent

    tool_name = validate_single_string_arg(tool_name, 'tool_name')

    if '-' in tool_name:
        warn('tool_name should not contain the "-" character. Reformatted name from:{} to '
             '{}'.format(tool_name, tool_name.replace('-', '_')))
    tool_name = tool_name.replace('-', '_')

    group_name = h5_main.name.split('/')[-1] + '-' + tool_name + '_'
    group_name = assign_group_index(h5_parent_group, group_name)

    h5_group = h5_parent_group.create_group(group_name)

    write_book_keeping_attrs(h5_group)

    # Also add some basic attributes like source and tool name. This will allow relaxation of nomenclature restrictions:
    # this are NOT being used right now but will be in the subsequent versions of pyNSID
    write_simple_attrs(h5_group, {'tool': tool_name, 'num_source_dsets': 1})
    # in this case, there is only one source
    if h5_parent_group.file == h5_main.file:
        for dset_ind, dset in enumerate([h5_main]):
            h5_group.attrs['source_' + '{:03d}'.format(dset_ind)] = dset.ref

    return h5_group


def find_results_groups(h5_main, tool_name, h5_parent_group=None):
    """
    Finds a list of all groups containing results of the process of name
    ``tool_name`` being applied to the dataset

    Parameters
    ----------
    h5_main : h5 dataset reference
        Reference to the target dataset to which the tool was applied
    tool_name : String / unicode
        Name of the tool applied to the target dataset
    h5_parent_group : h5py.Group, optional. Default = None
        Parent group under which the results group will be searched for. Use
        this option when the results groups are contained in different HDF5
        file compared to `h5_main`. BY default, this function will search
        within the same group that contains `h5_main`

    Returns
    -------
    groups : list of references to :class:`h5py.Group` objects
        groups whose name contains the tool name and the dataset name
    """
    warn('The behavior of find_results_group is very likely to change soon '
         'and significantly. Use this function with caution', FutureWarning)

    if not isinstance(h5_main, h5py.Dataset):
        raise TypeError('h5_main should be a h5py.Dataset object')
    tool_name = validate_single_string_arg(tool_name, 'tool_name')

    if h5_parent_group is not None:
        if not isinstance(h5_parent_group, (h5py.File, h5py.Group)):
            raise TypeError("'h5_parent_group' should either be a h5py.File "
                            "or h5py.Group object")
    else:
        h5_parent_group = h5_main.parent

    dset_name = h5_main.name.split('/')[-1]
    groups = []
    for key in h5_parent_group.keys():
        if dset_name in key and tool_name in key and isinstance(h5_parent_group[key], h5py.Group):
            groups.append(h5_parent_group[key])
    return groups


def check_for_old(h5_base, tool_name, new_parms=None, target_dset=None,
                  h5_parent_goup=None, verbose=False):
    """
    Check to see if the results of a tool already exist and if they
    were performed with the same parameters.

    Parameters
    ----------
    h5_base : h5py.Dataset object
           Dataset on which the tool is being applied to
    tool_name : str
           process or analysis name
    new_parms : dict, optional
           Parameters with which this tool will be performed.
    target_dset : str, optional, default = None
            Name of the dataset whose attributes will be compared against new_parms.
            Default - checking against the group
    h5_parent_goup : h5py.Group, optional. Default = None
            The group to search under. Use this option when `h5_base` and
            the potential results groups (within `h5_parent_goup` are located
            in different HDF5 files. Default - search within h5_base.parent
    verbose : bool, optional, default = False
           Whether or not to print debugging statements

    Returns
    -------
    group : list
           List of all :class:`h5py.Group` objects with parameters matching
           those in `new_parms`
    """
    warn('The behavior of check_for_old is very likely to change soon '
         '. Use this function with caution', FutureWarning)

    if not isinstance(h5_base, h5py.Dataset):
        raise TypeError('h5_base should be a h5py.Dataset object')
    tool_name = validate_single_string_arg(tool_name, 'tool_name')

    if h5_parent_goup is not None:
        if not isinstance(h5_parent_goup, (h5py.File, h5py.Group)):
            raise TypeError("'h5_parent_group' should either be a h5py.File "
                            "or h5py.Group object")
    else:
        h5_parent_goup = h5_base.parent

    if new_parms is None:
        new_parms = dict()
    else:
        if not isinstance(new_parms, dict):
            raise TypeError('new_parms should be a dict')
    if target_dset is not None:
        target_dset = validate_single_string_arg(target_dset, 'target_dset')

    matching_groups = []
    groups = find_results_groups(h5_base, tool_name,
                                 h5_parent_group=h5_parent_goup)

    for group in groups:
        if verbose:
            print('Looking at group - {}'.format(group.name.split('/')[-1]))

        h5_obj = group
        if target_dset is not None:
            if target_dset in group.keys():
                h5_obj = group[target_dset]
            else:
                if verbose:
                    print('{} did not contain the target dataset: {}'.format(group.name.split('/')[-1],
                                                                             target_dset))
                continue

        if check_for_matching_attrs(h5_obj, new_parms=new_parms, verbose=verbose):
            # return group
            matching_groups.append(group)

    return matching_groups


def check_for_matching_attrs(h5_obj, new_parms=None, verbose=False):
    """
    Compares attributes in the given H5 object against those in the provided
    dictionary and returns True if the parameters match, and False otherwise

    Parameters
    ----------
    h5_obj : h5py object (Dataset or :class:`h5py.Group`)
        Object whose attributes will be compared against ``new_parms``
    new_parms : dict, optional. default = empty dictionary
        Parameters to compare against the attributes present in h5_obj
    verbose : bool, optional, default = False
       Whether or not to print debugging statements

    Returns
    -------
    tests: bool
        Whether or not all paramters in new_parms matched with those in h5_obj's attributes
    """
    if not isinstance(h5_obj, (h5py.Dataset, h5py.Group, h5py.File)):
        raise TypeError('h5_obj should be a h5py.Dataset, h5py.Group, or h5py.File object')
    if new_parms is None:
        new_parms = dict()
    else:
        if not isinstance(new_parms, dict):
            raise TypeError('new_parms should be a dictionary')

    tests = []
    for key in new_parms.keys():

        if verbose:
            print('Looking for new attribute named: {}'.format(key))

        # HDF5 cannot store None as an attribute anyway. ignore
        if new_parms[key] is None:
            continue

        try:
            old_value = get_attr(h5_obj, key)
        except KeyError:
            # if parameter was not found assume that something has changed
            if verbose:
                print('New parm: {} \t- new parm not in group *****'.format(key))
            tests.append(False)
            break

        if isinstance(old_value, np.ndarray):
            if not isinstance(new_parms[key], Iterable):
                if verbose:
                    print('New parm: {} \t- new parm not iterable unlike old parm *****'.format(key))
                tests.append(False)
                break
            new_array = np.array(new_parms[key])
            if old_value.size != new_array.size:
                if verbose:
                    print('New parm: {} \t- are of different sizes ****'.format(key))
                tests.append(False)
            else:
                try:
                    answer = np.allclose(old_value, new_array)
                except TypeError:
                    # comes here when comparing string arrays
                    # Not sure of a better way
                    answer = []
                    for old_val, new_val in zip(old_value, new_array):
                        answer.append(old_val == new_val)
                    answer = np.all(answer)
                if verbose:
                    print('New parm: {} \t- match: {}'.format(key, answer))
                tests.append(answer)
        else:
            """if isinstance(new_parms[key], collections.Iterable):
                if verbose:
                    print('New parm: {} \t- new parm is iterable unlike old parm *****'.format(key))
                tests.append(False)
                break"""
            answer = np.all(new_parms[key] == old_value)
            if verbose:
                print('New parm: {} \t- match: {}'.format(key, answer))
            tests.append(answer)
    if verbose:
        print('')

    return all(tests)


def get_source_dataset(h5_group):
    """
    Find the name of the source dataset used to create the input `h5_group`,
    so long as the source dataset is in the same HDF5 file
    Parameters
    ----------
    h5_group : :class:`h5py.Group`
        Child group whose source dataset will be returned
    Returns
    -------
    h5_source : NSIDataset object
        Main dataset from which this group was generated
    """
    if not isinstance(h5_group, h5py.Group):
        raise TypeError('h5_group should be a h5py.Group object')

    h5_parent_group = h5_group.parent
    group_name = h5_group.name.split('/')[-1]
    # What if the group name was not formatted according to Pycroscopy rules?
    name_split = group_name.split('-')
    if len(name_split) != 2:
        raise ValueError("The provided group's name could not be split by '-' as expected in "
                         "SourceDataset-ProcessName_000")
    h5_source = h5_parent_group[name_split[0]]

    if not isinstance(h5_source, h5py.Dataset):
        raise ValueError('Source object was not a dataset!')

    return h5_source
