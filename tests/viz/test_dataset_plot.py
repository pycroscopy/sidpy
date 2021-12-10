"""
Created on Thurs Dec 10 2021

@author: Gerd Duscher
"""

import unittest
import sys
import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
mpl.use('Agg')
import numpy as np
sys.path.insert(0, "../../sidpy/")
import sidpy


class TestDatasetPlot(unittest.TestCase):
    def test_spectrum(self):
        x = np.random.normal(3, 2.5, size=1024)
        dset = sidpy.Dataset.from_array(x)

        # dataset metadata
        dset.data_type = 'spectrum'
        dset.title = 'random'
        dset.quantity = 'intensity'
        dset.units = 'a.u.'

        # dimension with metadata
        scale = .5
        offset = 390
        dset.set_dimension(0, sidpy.Dimension(np.arange(dset.shape[0]) * scale + offset, 'energy'))
        dset.dim_0.dimension_type = 'spectral'
        dset.energy.units = 'eV'
        dset.energy.quantity = 'energy'

        view = dset.plot()

        x_y = view.axes[0].lines[0].get_xydata()
        self.assertTrue(np.allclose(x_y[:, 1], x))
        self.assertTrue(np.allclose(x_y[:, 0], dset.energy))
        self.assertEqual(dset.title, view.axes[0].get_title())
        self.assertEqual(f"{dset.energy.quantity} ({dset.energy.units})", view.axes[0].get_xlabel())

    def test_image(self):
        x = np.random.normal(3, 2.5, size=(512, 512))
        dset = sidpy.Dataset.from_array(x)
        dset.data_type = 'image'
        dset.units = 'counts'
        dset.quantity = 'intensity'
        dset.title = 'random'
        dset.set_dimension(0, sidpy.Dimension(np.arange(dset.shape[0]) * .02, 'x'))
        dset.x.dimension_type = 'spatial'
        dset.x.units = 'nm'
        dset.x.quantity = 'distance'
        dset.set_dimension(1, sidpy.Dimension(np.arange(dset.shape[1]) * .02, 'y'))
        dset.y.dimension_type = 'spatial'
        dset.y.units = 'nm'
        dset.y.quantity = 'distance'
        view = dset.plot()

        data = view.axes[0].images[0].get_array().data

        self.assertTrue(np.allclose(data.shape, x.shape))

        self.assertEqual(dset.title, view.axes[0].get_title())
        self.assertEqual(f"{dset.x.quantity} ({dset.x.units})", view.axes[0].get_xlabel())


if __name__ == '__main__':
    unittest.main()
