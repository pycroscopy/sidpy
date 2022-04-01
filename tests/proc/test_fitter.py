import sys
import unittest
import numpy as np

sys.path.insert(0, "../../sidpy/")

from sidpy.proc.fitter import SidFitter
from sidpy import Dataset, Dimension, DataType
import sidpy as sid
from ..sid.test_dataset import validate_dataset_properties
import inspect


class Test_3D_dset_1Dfit(unittest.TestCase):
    def setUp(self):
        self.data_set_3D, self.xvec = make_3D_dataset(shape=(10, 10, 32))
        fitter = SidFitter(self.data_set_3D, xvec=self.xvec,
                           fit_fn=return_quad, guess_fn=guess_quad, num_workers=8,
                           threads=2, return_cov=True, return_fit=True, return_std=True,
                           km_guess=True, n_clus=10)

        lb, ub = [-50, -50, -50], [50, 50, 50]
        self.fit_results = fitter.do_fit(bounds=(lb, ub), maxfev=100)

        self.metadata = self.data_set_3D.metadata.copy()
        fit_parms_dict = {'fit_parameters_labels': None,
                          'fitting_function': inspect.getsource(return_quad),
                          'guess_function': inspect.getsource(guess_quad),
                          'ind_dims': (0, 1)
                          }
        self.metadata['fit_parms_dict'] = fit_parms_dict

    def test_num_fitter_outputs(self):
        self.assertEqual(len(self.fit_results), 4)  # add assertion here

    def test_fit_parms_dset(self):
        # First dataset would be the fitting parameters dataset
        self.assertEqual(self.fit_results[0].shape, (10, 10, 3))
        ## Getting the dimension dict
        dim_dict = {0: self.data_set_3D._axes[0].copy(), 1: self.data_set_3D._axes[1].copy(),
                    2: Dimension(np.arange(3),
                                 name='fit_parms', units='a.u.',
                                 quantity='fit_parameters',
                                 dimension_type='temporal')}

        validate_dataset_properties(self, self.fit_results[0], np.array(self.fit_results[0]),
                                    title='Fitting_Map', data_type=DataType.IMAGE_STACK,
                                    dimension_dict=dim_dict,
                                    original_metadata=self.data_set_3D.original_metadata.copy(),
                                    metadata=self.metadata)

    def test_cov_dset(self):
        # Second dataset is the covariance dataset
        self.assertEqual(self.fit_results[1].shape, (10, 10, 3, 3))
        ## Getting the dim_dict
        dim_dict = {0: self.data_set_3D._axes[0].copy(), 1: self.data_set_3D._axes[1].copy(),
                    2: Dimension(np.arange(3),
                                 name='fit_cov_parms_x', units='a.u.',
                                 quantity='fit_cov_parameters',
                                 dimension_type='spectral'),
                    3: Dimension(np.arange(3),
                                 name='fit_cov_parms_y', units='a.u.',
                                 quantity='fit_cov_parameters',
                                 dimension_type='spectral')
                    }

        validate_dataset_properties(self, self.fit_results[1], np.array(self.fit_results[1]),
                                    title='Fitting_Map_Covariance', data_type=DataType.IMAGE_4D,
                                    dimension_dict=dim_dict,
                                    original_metadata=self.data_set_3D.original_metadata.copy(),
                                    metadata=self.metadata)

    def test_std_dev_dset(self):
        # Third dataset is the std_dev dataset
        self.assertEqual(self.fit_results[2].shape, (10, 10, 3))
        ## Getting the dim_dict
        dim_dict = {0: self.data_set_3D._axes[0].copy(), 1: self.data_set_3D._axes[1].copy(),
                    2: Dimension(np.arange(3),
                                 name='std_dev', units='a.u.',
                                 quantity='std_dev_fit_parms',
                                 dimension_type='temporal')}
        validate_dataset_properties(self, self.fit_results[2], np.array(self.fit_results[2]),
                                    title='Fitting_Map_std_dev', data_type=DataType.IMAGE_STACK,
                                    dimension_dict=dim_dict,
                                    original_metadata=self.data_set_3D.original_metadata.copy(),
                                    metadata=self.metadata)

    def test_fitted_dset(self):
        # Fourth dataset is the fitted dataset
        self.assertEqual(self.fit_results[3].shape, self.data_set_3D.shape)
        validate_dataset_properties(self, self.fit_results[3], np.array(self.fit_results[3]),
                                    title='fitted_' + self.data_set_3D.title, data_type=self.data_set_3D.data_type,
                                    quantity=self.data_set_3D.quantity, modality=self.data_set_3D.modality,
                                    units=self.data_set_3D.units, source=self.data_set_3D.source,
                                    dimension_dict=self.data_set_3D._axes,
                                    original_metadata=self.data_set_3D.original_metadata,
                                    metadata=self.metadata)




def return_quad(x, *parms):
    a, b, c, = parms
    return a * x ** 2 + b * x + c


def guess_quad(xvec, yvec):
    popt = np.polyfit(xvec, yvec, 2)
    return popt


def make_3D_dataset(shape=(60, 60, 32), cycles=1):
    data_mat = np.zeros(shape=(shape[0], shape[1], shape[2], cycles))
    xvec = np.linspace(0, 5, shape[-1])

    for row in range(data_mat.shape[0]):
        for col in range(data_mat.shape[1]):
            for cycle in range(cycles):
                if row ** 2 + col ** 2 < data_mat.shape[0] * data_mat.shape[1] / 3:
                    a = np.random.normal(loc=3.0, scale=0.4)
                    b = np.random.normal(loc=1.0, scale=2.4)
                    c = np.random.normal(loc=0.5, scale=1.0)
                else:
                    a = np.random.normal(loc=1.0, scale=0.8)
                    b = np.random.normal(loc=0.0, scale=1.4)
                    c = np.random.normal(loc=-0.5, scale=1.3)
                data_mat[row, col, :, cycle] = return_quad(xvec, *[a, b, c]) + 2.5 * np.random.normal(size=len(xvec))

    data_mat = np.squeeze(data_mat)
    parms_dict = {'info_1': np.linspace(0, 1, 100), 'instrument': 'perseverence rover AFM'}

    # Let's convert it to a sidpy dataset

    # Specify dimensions
    x_dim = np.linspace(0, 1E-6,
                        data_mat.shape[0])
    y_dim = np.linspace(0, 1E-6,
                        data_mat.shape[1])

    z_dim = xvec

    # Make a sidpy dataset
    data_set = sid.Dataset.from_array(data_mat, title='Current_spectral_map')

    # Set the data type
    data_set.data_type = sid.DataType.SPECTRAL_IMAGE

    # Add quantity and units
    data_set.units = 'nA'
    data_set.quantity = 'Current'

    # Add dimension info
    data_set.set_dimension(0, sid.Dimension(x_dim,
                                            name='x',
                                            units='m', quantity='x',
                                            dimension_type='spatial'))
    data_set.set_dimension(1, sid.Dimension(y_dim,
                                            name='y',
                                            units='m', quantity='y',
                                            dimension_type='spatial'))

    data_set.set_dimension(2, sid.Dimension(z_dim,
                                            name='Voltage',
                                            units='V', quantity='Voltage',
                                            dimension_type='spectral'))
    if cycles > 1:
        cycles_dim = np.arange(cycles)
        data_set.set_dimension(3, sid.Dimension(cycles_dim,
                                                name='Cycle',
                                                units='#', quantity='Cycle',
                                                dimension_type='spectral'))

    # append metadata
    data_set.metadata = parms_dict

    return data_set, xvec


if __name__ == '__main__':
    unittest.main()
