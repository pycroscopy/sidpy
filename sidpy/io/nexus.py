"""Utilities for converting between sidpy datasets and NeXus HDF5."""

from __future__ import absolute_import, division, print_function, unicode_literals

import datetime
import json

import h5py
import numpy as np

from sidpy.sid import Dataset, Dimension

__all__ = ["sidpy_to_nexus_hdf5", "nexus_to_sidpy"]


def _clean_name(name, fallback):
    if name is None:
        name = ""
    name = str(name).strip().replace("/", "_")
    return name if name else fallback


def _ensure_unique_name(name, used_names):
    if name not in used_names:
        used_names.add(name)
        return name

    index = 1
    while True:
        candidate = "{}_{}".format(name, index)
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        index += 1


def _decode_if_bytes(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [_decode_if_bytes(item) for item in value.tolist()]
    return value


def _normalize_axes_attr(value):
    value = _decode_if_bytes(value)
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return None


def _json_ready(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(_json_ready(key)): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json_dataset(parent, name, payload):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    parent.create_dataset(name, data=json.dumps(_json_ready(payload)), dtype=string_dtype)


def _read_json_dataset(parent, name):
    if name not in parent:
        return {}
    raw = parent[name][()]
    raw = _decode_if_bytes(raw)
    if not raw:
        return {}
    return json.loads(raw)


def _set_root_attrs(h5_file, default_entry):
    time_stamp = datetime.datetime.now().isoformat()
    h5_file.attrs["default"] = default_entry
    h5_file.attrs["file_name"] = h5_file.filename
    h5_file.attrs["file_time"] = time_stamp
    h5_file.attrs["creator"] = "sidpy"
    h5_file.attrs["HDF5_Version"] = h5py.version.hdf5_version
    h5_file.attrs["h5py_version"] = h5py.version.version


def sidpy_to_nexus_hdf5(dataset, h5_path, entry_name="entry", nxdata_name="data",
                        signal_name="data", mode="w", compression=None):
    """
    Write a sidpy.Dataset to a NeXus-compatible HDF5 file.

    Parameters
    ----------
    dataset : sidpy.Dataset
        Dataset to serialize.
    h5_path : str or h5py.File
        Destination HDF5 path or open file handle.
    entry_name : str, optional
        Name of the NXentry group.
    nxdata_name : str, optional
        Name of the NXdata group.
    signal_name : str, optional
        Name of the primary signal dataset within NXdata.
    mode : str, optional
        File mode used when `h5_path` is a path.
    compression : str, optional
        Compression passed to h5py when creating datasets.

    Returns
    -------
    str or h5py.Dataset
        Signal dataset path for path-based writes, or the written h5py.Dataset
        when an open file handle is provided.
    """
    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be a sidpy.Dataset")

    entry_name = _clean_name(entry_name, "entry")
    nxdata_name = _clean_name(nxdata_name, "data")
    signal_name = _clean_name(signal_name, "data")

    close_file = False
    if isinstance(h5_path, h5py.File):
        h5_file = h5_path
    else:
        h5_file = h5py.File(h5_path, mode)
        close_file = True

    try:
        if entry_name in h5_file:
            del h5_file[entry_name]

        _set_root_attrs(h5_file, entry_name)

        entry = h5_file.create_group(entry_name)
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["default"] = nxdata_name
        entry.create_dataset("title", data=dataset.title or signal_name)

        nxdata = entry.create_group(nxdata_name)
        nxdata.attrs["NX_class"] = "NXdata"
        nxdata.attrs["signal"] = signal_name

        signal_kwargs = {}
        if compression is not None:
            signal_kwargs["compression"] = compression

        signal = nxdata.create_dataset(signal_name, data=np.array(dataset), **signal_kwargs)
        signal.attrs["units"] = dataset.units
        signal.attrs["quantity"] = dataset.quantity
        signal.attrs["data_type"] = dataset.data_type.name
        signal.attrs["modality"] = dataset.modality
        signal.attrs["source"] = dataset.source
        signal.attrs["title"] = dataset.title
        signal.attrs["long_name"] = dataset.data_descriptor

        used_names = {signal_name}
        axes_names = []
        for dim_index in range(dataset.ndim):
            axis = dataset._axes.get(dim_index)
            if axis is None:
                axis_name = "."
                axes_names.append(axis_name)
                continue

            axis_name = _ensure_unique_name(_clean_name(axis.name, "dim_{}".format(dim_index)), used_names)
            axis_dset = nxdata.create_dataset(axis_name, data=np.asarray(axis.values))
            axis_dset.attrs["units"] = axis.units
            axis_dset.attrs["quantity"] = axis.quantity
            axis_dset.attrs["dimension_type"] = axis.dimension_type.name
            axis_dset.attrs["long_name"] = "{} ({})".format(axis.quantity, axis.units)
            nxdata.attrs["{}_indices".format(axis_name)] = dim_index
            axes_names.append(axis_name)

        nxdata.attrs["axes"] = np.asarray(axes_names, dtype=h5py.string_dtype(encoding="utf-8"))

        if dataset.variance is not None:
            nxdata.create_dataset("{}_errors".format(signal_name), data=np.array(dataset.variance))

        sidpy_collection = entry.create_group("sidpy_metadata")
        sidpy_collection.attrs["NX_class"] = "NXcollection"
        _write_json_dataset(sidpy_collection, "metadata", dataset.metadata)
        _write_json_dataset(sidpy_collection, "original_metadata", dataset.original_metadata)
        _write_json_dataset(sidpy_collection, "provenance", dataset.provenance)

        h5_file.flush()

        if close_file:
            return signal.name
        return signal
    finally:
        if close_file:
            h5_file.close()


def _resolve_default_child(parent, default_name):
    if default_name is None:
        return None

    default_name = _decode_if_bytes(default_name)
    if default_name in parent:
        return parent[default_name]

    if isinstance(default_name, str):
        if default_name.startswith("/"):
            return parent.file[default_name]
        candidate = "{}/{}".format(parent.name.rstrip("/"), default_name).replace("//", "/")
        if candidate in parent.file:
            return parent.file[candidate]
    return None


def _find_nxentry(h5_file):
    default_entry = _resolve_default_child(h5_file, h5_file.attrs.get("default"))
    if isinstance(default_entry, h5py.Group) and _decode_if_bytes(default_entry.attrs.get("NX_class")) == "NXentry":
        return default_entry

    for key in h5_file:
        obj = h5_file[key]
        if isinstance(obj, h5py.Group) and _decode_if_bytes(obj.attrs.get("NX_class")) == "NXentry":
            return obj
    raise ValueError("Could not find an NXentry group in the provided file")


def _find_nxdata(entry):
    default_nxdata = _resolve_default_child(entry, entry.attrs.get("default"))
    if isinstance(default_nxdata, h5py.Group) and _decode_if_bytes(default_nxdata.attrs.get("NX_class")) == "NXdata":
        return default_nxdata

    for key in entry:
        obj = entry[key]
        if isinstance(obj, h5py.Group) and _decode_if_bytes(obj.attrs.get("NX_class")) == "NXdata":
            return obj
    raise ValueError("Could not find an NXdata group in the provided entry")


def nexus_to_sidpy(h5_path, entry_path=None, nxdata_path=None, signal_name=None):
    """
    Read a NeXus HDF5 NXdata signal into a sidpy.Dataset.

    Parameters
    ----------
    h5_path : str or h5py.File
        Source HDF5 file path or open file handle.
    entry_path : str, optional
        Explicit path to the NXentry group.
    nxdata_path : str, optional
        Explicit path to the NXdata group.
    signal_name : str, optional
        Explicit name of the signal dataset inside NXdata.

    Returns
    -------
    sidpy.Dataset
        Restored dataset.
    """
    if isinstance(h5_path, h5py.File):
        h5_file = h5_path
    else:
        h5_file = h5py.File(h5_path, "r")

    if nxdata_path is not None:
        nxdata = h5_file[nxdata_path]
        if not isinstance(nxdata, h5py.Group):
            raise TypeError("nxdata_path must point to a group")
        if entry_path is None:
            entry = nxdata.parent
        else:
            entry = h5_file[entry_path]
    else:
        if entry_path is not None:
            entry = h5_file[entry_path]
        else:
            entry = _find_nxentry(h5_file)
        nxdata = _find_nxdata(entry)

    if _decode_if_bytes(nxdata.attrs.get("NX_class")) != "NXdata":
        raise ValueError("The selected group is not an NXdata group")

    if signal_name is None:
        signal_name = _decode_if_bytes(nxdata.attrs.get("signal"))
    signal_name = _clean_name(signal_name, "data")

    if signal_name not in nxdata:
        raise ValueError("Could not find signal dataset '{}' in NXdata".format(signal_name))

    signal = nxdata[signal_name]
    entry_title = _decode_if_bytes(entry["title"][()]) if "title" in entry else signal_name
    signal_title = _decode_if_bytes(signal.attrs.get("title", ""))
    dataset = Dataset.from_array(np.array(signal), title=signal_title or entry_title)

    dataset.units = _decode_if_bytes(signal.attrs.get("units", "generic"))
    dataset.quantity = _decode_if_bytes(signal.attrs.get("quantity", "generic"))

    data_type = _decode_if_bytes(signal.attrs.get("data_type", "UNKNOWN"))
    try:
        dataset.data_type = data_type
    except Warning:
        dataset.data_type = "UNKNOWN"

    dataset.modality = _decode_if_bytes(signal.attrs.get("modality", "generic"))
    dataset.source = _decode_if_bytes(signal.attrs.get("source", "generic"))
    dataset.title = signal_title or entry_title

    axes_names = _normalize_axes_attr(nxdata.attrs.get("axes"))
    if axes_names is None:
        axes_names = ["dim_{}".format(index) for index in range(dataset.ndim)]

    if len(axes_names) != dataset.ndim:
        raise ValueError("NXdata axes metadata does not match signal rank")

    for dim_index, axis_name in enumerate(axes_names):
        if axis_name == ".":
            continue
        if axis_name not in nxdata:
            continue

        axis_dset = nxdata[axis_name]
        axis_values = np.asarray(axis_dset[()])
        if axis_values.ndim != 1:
            raise NotImplementedError("Only 1D NXdata axes are currently supported")
        if axis_values.shape[0] != dataset.shape[dim_index]:
            raise ValueError("Axis '{}' length does not match data dimension {}".format(axis_name, dim_index))

        dimension = Dimension(axis_values,
                              name=axis_name,
                              quantity=_decode_if_bytes(axis_dset.attrs.get("quantity", axis_name)),
                              units=_decode_if_bytes(axis_dset.attrs.get("units", "generic")),
                              dimension_type=_decode_if_bytes(axis_dset.attrs.get("dimension_type", "UNKNOWN")))
        dataset.set_dimension(dim_index, dimension)

    if "sidpy_metadata" in entry:
        sidpy_collection = entry["sidpy_metadata"]
        if isinstance(sidpy_collection, h5py.Group):
            dataset.metadata = _read_json_dataset(sidpy_collection, "metadata")
            dataset.original_metadata = _read_json_dataset(sidpy_collection, "original_metadata")
            provenance = _read_json_dataset(sidpy_collection, "provenance")
            if provenance:
                dataset.provenance = provenance

    dataset.h5_dataset = signal
    return dataset
