# -*- coding: utf-8 -*-
"""Tests for NeXus HDF5 conversion helpers."""

from __future__ import division, print_function, unicode_literals, absolute_import

import os
import tempfile
import unittest

import h5py
import numpy as np

import sidpy


def _decode_attr_list(values):
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode('utf-8'))
        else:
            decoded.append(value)
    return decoded


class TestNexusRoundTrip(unittest.TestCase):

    def setUp(self):
        self.data = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.dataset = sidpy.Dataset.from_array(self.data, title='test_image')
        self.dataset.units = 'counts'
        self.dataset.quantity = 'intensity'
        self.dataset.data_type = 'image'
        self.dataset.modality = 'STEM'
        self.dataset.source = 'unit-test'
        self.dataset.set_dimension(0, sidpy.Dimension(np.linspace(0, 2, 3),
                                                      name='row',
                                                      quantity='distance',
                                                      units='nm',
                                                      dimension_type='spatial'))
        self.dataset.set_dimension(1, sidpy.Dimension(np.linspace(0, 3, 4),
                                                      name='col',
                                                      quantity='distance',
                                                      units='nm',
                                                      dimension_type='spatial'))
        self.dataset.metadata = {'experiment': 'demo', 'values': [1, 2, 3]}
        self.dataset.original_metadata = {'vendor': {'name': 'acme'}}
        self.dataset.provenance = {'sidpy': {'creator': 'test'}}

    def test_sidpy_to_nexus_hdf5(self):
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            file_path = tmp.name

        try:
            signal_path = sidpy.sidpy_to_nexus_hdf5(self.dataset, file_path)
            self.assertEqual(signal_path, '/entry/data/data')

            with h5py.File(file_path, 'r') as h5_file:
                self.assertEqual(h5_file.attrs['default'], 'entry')
                self.assertEqual(h5_file['entry'].attrs['NX_class'], 'NXentry')
                self.assertEqual(h5_file['entry'].attrs['default'], 'data')
                self.assertEqual(h5_file['entry/data'].attrs['NX_class'], 'NXdata')
                self.assertEqual(h5_file['entry/data'].attrs['signal'], 'data')
                self.assertEqual(_decode_attr_list(list(h5_file['entry/data'].attrs['axes'])), ['row', 'col'])
                np.testing.assert_allclose(h5_file['entry/data/data'][()], self.data)
                np.testing.assert_allclose(h5_file['entry/data/row'][()], self.dataset.row.values)
                np.testing.assert_allclose(h5_file['entry/data/col'][()], self.dataset.col.values)
        finally:
            os.remove(file_path)

    def test_nexus_to_sidpy(self):
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            file_path = tmp.name

        restored = None
        try:
            sidpy.sidpy_to_nexus_hdf5(self.dataset, file_path)
            restored = sidpy.nexus_to_sidpy(file_path)

            self.assertIsInstance(restored, sidpy.Dataset)
            np.testing.assert_allclose(np.array(restored), self.data)
            self.assertEqual(restored.title, self.dataset.title)
            self.assertEqual(restored.units, self.dataset.units)
            self.assertEqual(restored.quantity, self.dataset.quantity)
            self.assertEqual(restored.data_type, self.dataset.data_type)
            self.assertEqual(restored.modality, self.dataset.modality)
            self.assertEqual(restored.source, self.dataset.source)
            self.assertEqual(restored.metadata, self.dataset.metadata)
            self.assertEqual(restored.original_metadata, self.dataset.original_metadata)
            self.assertEqual(restored.provenance, self.dataset.provenance)
            np.testing.assert_allclose(restored.row.values, self.dataset.row.values)
            np.testing.assert_allclose(restored.col.values, self.dataset.col.values)
            self.assertEqual(restored.row.units, 'nm')
            self.assertEqual(restored.col.units, 'nm')
        finally:
            if restored is not None and restored.h5_dataset is not None:
                restored.h5_dataset.file.close()
            os.remove(file_path)

    def test_bytes_metadata_round_trip(self):
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            file_path = tmp.name

        restored = None
        try:
            self.dataset.original_metadata = {b'bname': b'image.bin'}
            sidpy.sidpy_to_nexus_hdf5(self.dataset, file_path)
            restored = sidpy.nexus_to_sidpy(file_path)
            self.assertEqual(restored.original_metadata, {'bname': 'image.bin'})
        finally:
            if restored is not None and restored.h5_dataset is not None:
                restored.h5_dataset.file.close()
            os.remove(file_path)


if __name__ == '__main__':
    unittest.main()
