import unittest

import numpy as np
import sidpy as sid
import tempfile
from pathlib import Path
import logging
from urllib.request import urlretrieve

from sidpy.proc.fitter_refactor import SidpyFitter

from .fitter_function_utils import (
    gaussian_2d,
    gaussian_2d_guess,
    loop_fit_function,
    generate_guess,
    SHO_fit_flattened,
    sho_guess_fn,
)

log = logging.getLogger(__name__)


class TestSidpyFitterIntegration(unittest.TestCase):
    def test_beps_fit(self):
        try:
            import SciFiReaders as sr
        except Exception as exc:
            self.skipTest(f"SciFiReaders unavailable in this environment: {exc}")

        with tempfile.TemporaryDirectory() as d:
            tmp_path = Path(d)

            url = "https://github.com/pycroscopy/DTMicroscope/raw/refs/heads/main/data/AFM/BEPS_PTO_50x50.h5"
            beps_file = tmp_path / "BEPS_PTO_50x50.h5"

            urlretrieve(url, beps_file)
            self.assertTrue(beps_file.exists())

            reader = sr.NSIDReader(str(beps_file))
            data = reader.read()['Channel_000']

            data_loop_cropped = data[10:20, 10:20, :]

            for ind_x in range(data_loop_cropped.shape[0]):
                for ind_y in range(data_loop_cropped.shape[1]):
                    data_loop_cropped[ind_x, ind_y, :] = np.roll(
                        np.array(data_loop_cropped[ind_x, ind_y, :]), 16
                    ) * 1E3

            dc_vec = data._axes[2].values
            dc_vec_rolled = np.roll(dc_vec, 16)
            data_loop_cropped.set_dimension(
                2,
                sid.Dimension(
                    dc_vec_rolled,
                    name='DC Offset',
                    quantity='Voltage',
                    units='Volts',
                    dimension_type='spectral',
                ),
            )

            fitter = SidpyFitter(data_loop_cropped, loop_fit_function, generate_guess, ind_dims=(2,))
            fitter.setup_calc()

            fitter.do_guess()
            result_beps = fitter.do_fit(use_kmeans=False, n_clusters=12)
            result_beps = fitter.do_fit(use_kmeans=True, n_clusters=6, return_cov=True, loss='huber')

            model_source = result_beps[0].metadata['source_code']['model_function']
            from scipy.special import erf

            reloaded_model = fitter.reconstruct_function(model_source, context={'erf': erf})

            vdc = data_loop_cropped._axes[2].values
            raw_loop = data_loop_cropped[4, 3, :]
            fit_loop = reloaded_model(vdc, *np.array(data_loop_cropped[4, 3, :]))
            assert np.isfinite((fit_loop - raw_loop).mean())

    def test_sho_fit(self):
        try:
            import SciFiReaders as sr
        except Exception as exc:
            self.skipTest(f"SciFiReaders unavailable in this environment: {exc}")

        with tempfile.TemporaryDirectory() as d:
            tmp_path = Path(d)

            url = "https://www.dropbox.com/scl/fi/wmablmstf3gw0dokzen6o/PTO_60x60_3rdarea_0003.h5?rlkey=ozr89y9ztggznj2p7fjls0zz2&dl=1"
            sho_dataset = tmp_path / 'PTO_60x60_3rdarea_0003.h5'

            urlretrieve(url, sho_dataset)
            self.assertTrue(sho_dataset.exists())

            reader = sr.Usid_reader(str(sho_dataset))
            data_sho = reader.read()[0]

            freq_axis = data_sho.labels.index('Frequency (Hz)')
            data_sho_cropped = data_sho[:10, :10, :, :, :]

            fitter_sho = SidpyFitter(data_sho_cropped, SHO_fit_flattened, sho_guess_fn, ind_dims=(2,))
            fitter_sho.setup_calc()
            fitter_sho.do_guess()
            fitter_sho.do_fit(use_kmeans=False)
            fitter_sho.do_fit(use_kmeans=True, n_clusters=10)


class TestSidpyFitter1D(unittest.TestCase):
    def setUp(self):
        self.data_set, self.xvec = make_3D_dataset(shape=(3, 4, 7))
        self.fitter = SidpyFitter(
            self.data_set,
            model_function=return_quad,
            guess_function=guess_quad,
            ind_dims=(2,),
        )
        self.fitter.setup_calc()
        self.fit_params, self.fit_cov = self.fitter.do_fit(return_cov=True)

    def test_fit_parameter_map(self):
        self.assertIsInstance(self.fit_params, sid.Dataset)
        self.assertEqual(self.fit_params.shape, (3, 4, 3))
        self.assertTrue(np.all(np.isfinite(np.array(self.fit_params))))
        self.assertEqual(self.fit_params.metadata["fit_parms_dict"]["ind_dims"], (2,))

    def test_covariance_map(self):
        self.assertIsInstance(self.fit_cov, sid.Dataset)
        self.assertEqual(self.fit_cov.shape, (3, 4, 3, 3))
        self.assertTrue(np.all(np.isfinite(np.array(self.fit_cov))))
        self.assertEqual(self.fit_cov.metadata["fit_parms_dict"]["ind_dims"], (2,))


class TestSidpyFitter2D(unittest.TestCase):
    def setUp(self):
        self.data_set, self.xyvec = make_4D_dataset(shape=(3, 5, 9, 13))
        self.fitter = SidpyFitter(
            self.data_set,
            model_function=gauss_2D,
            guess_function=gauss_2D_guess,
            ind_dims=(2, 3),
        )
        self.fitter.setup_calc()
        self.fit_params, self.fit_cov = self.fitter.do_fit(
            return_cov=True,
            fit_parameter_labels=["amplitude", "x", "y", "sigma", "offset"],
        )

    def test_fit_parameter_map(self):
        self.assertIsInstance(self.fit_params, sid.Dataset)
        self.assertEqual(self.fit_params.shape, (3, 5, 5))
        self.assertEqual(self.fit_params.metadata["fit_parms_dict"]["fit_parameters_labels"],
                         ["amplitude", "x", "y", "sigma", "offset"])

    def test_covariance_map(self):
        self.assertIsInstance(self.fit_cov, sid.Dataset)
        self.assertEqual(self.fit_cov.shape, (3, 5, 5, 5))
        self.assertEqual(self.fit_cov.metadata["fit_parms_dict"]["fit_parameters_labels"],
                         ["amplitude", "x", "y", "sigma", "offset"])


def return_quad(x, *parms):
    a, b, c = parms
    return a * x ** 2 + b * x + c


def guess_quad(xvec, yvec):
    return np.polyfit(xvec, yvec, 2)


def make_3D_dataset(shape=(60, 60, 32), cycles=1):
    rng = np.random.default_rng(0)
    data_mat = np.zeros(shape=(shape[0], shape[1], shape[2], cycles))
    xvec = np.linspace(0, 5, shape[-1])

    for row in range(data_mat.shape[0]):
        for col in range(data_mat.shape[1]):
            for cycle in range(cycles):
                if row ** 2 + col ** 2 < data_mat.shape[0] * data_mat.shape[1] / 3:
                    a = rng.normal(loc=3.0, scale=0.4)
                    b = rng.normal(loc=1.0, scale=2.4)
                    c = rng.normal(loc=0.5, scale=1.0)
                else:
                    a = rng.normal(loc=1.0, scale=0.8)
                    b = rng.normal(loc=0.0, scale=1.4)
                    c = rng.normal(loc=-0.5, scale=1.3)
                data_mat[row, col, :, cycle] = return_quad(xvec, *[a, b, c]) + 2.5 * rng.normal(size=len(xvec))

    data_mat = np.squeeze(data_mat)
    parms_dict = {'info_1': np.linspace(0, 1, 100), 'instrument': 'perseverence rover AFM'}

    x_dim = np.linspace(0, 1E-6, data_mat.shape[0])
    y_dim = np.linspace(0, 1E-6, data_mat.shape[1])
    z_dim = xvec

    data_set = sid.Dataset.from_array(data_mat, title='Current_spectral_map')
    data_set.data_type = sid.DataType.SPECTRAL_IMAGE
    data_set.units = 'nA'
    data_set.quantity = 'Current'
    data_set.set_dimension(0, sid.Dimension(x_dim, name='x', units='m', quantity='x', dimension_type='spatial'))
    data_set.set_dimension(1, sid.Dimension(y_dim, name='y', units='m', quantity='y', dimension_type='spatial'))
    data_set.set_dimension(2, sid.Dimension(z_dim, name='Voltage', units='V', quantity='Voltage',
                                            dimension_type='spectral'))
    if cycles > 1:
        cycles_dim = np.arange(cycles)
        data_set.set_dimension(3, sid.Dimension(cycles_dim, name='Cycle', units='#', quantity='Cycle',
                                                dimension_type='spectral'))

    data_set.metadata = parms_dict
    return data_set, xvec


def gauss_2D(fitting_space, *parms):
    x = fitting_space[0]
    y = fitting_space[1]
    amplitude, xo, yo, sigma, offset = parms
    xo = float(xo)
    yo = float(yo)
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
    r = ((x_grid - xo) ** 2 + (y_grid - yo) ** 2) ** 0.5
    g = amplitude * np.exp(-(r / sigma) ** 2) + offset
    return g.ravel()


def gauss_2D_guess(fitting_space, data_flat):
    x = fitting_space[0]
    y = fitting_space[1]
    shape = (len(x), len(y))
    data_2d = np.array(data_flat).reshape(shape)

    offset = np.min(data_2d)
    amplitude = np.max(data_2d) - offset
    flat_idx = np.argmax(data_2d)
    idx_x, idx_y = np.unravel_index(flat_idx, shape)
    xo = x[idx_x]
    yo = y[idx_y]
    sigma = max((x[-1] - x[0]) / 6.0, (y[-1] - y[0]) / 6.0)
    return [amplitude, xo, yo, sigma, offset]


def make_4D_dataset(shape=(32, 16, 64, 48)):
    rng = np.random.default_rng(1)
    dataset = np.zeros(shape=shape, dtype=np.float64)
    xlen, ylen, kxlen, kylen = shape
    kx_vec = np.linspace(0, 11, kxlen)
    ky_vec = np.linspace(0, 7, kylen)
    kx, ky = np.meshgrid(kx_vec, ky_vec, indexing='ij')

    for row in range(xlen):
        for col in range(ylen):
            amp = np.sqrt(row * col // xlen * ylen) + 3.5
            sigma = rng.normal(loc=2.5, scale=0.34)
            offset = 0.1
            xo = rng.uniform(low=0, high=6)
            yo = rng.uniform(low=0, high=5)
            cur_parms = [amp, xo, yo, sigma, offset]
            gauss_mat = gauss_2D([kx_vec, ky_vec], *cur_parms)
            gauss_mat += np.mean(gauss_mat) / 2 * rng.normal(size=len(gauss_mat))
            dataset[row, col, :, :] = gauss_mat.reshape([kxlen, kylen])

    parms_dict = {'info_1': np.linspace(0, 5.6, 30), 'instrument': 'opportunity rover AFM'}
    x_dim = np.linspace(0, 1E-6, dataset.shape[0])
    y_dim = np.linspace(0, 2E-6, dataset.shape[1])
    kx_dim = np.linspace(0, 11, kxlen)
    ky_dim = np.linspace(0, 5, kylen)

    data_set = sid.Dataset.from_array(dataset, name='4D_STEM')
    data_set.data_type = sid.DataType.IMAGE_4D
    data_set.units = 'nA'
    data_set.quantity = 'Current'
    data_set.set_dimension(0, sid.Dimension(x_dim, name='x', units='m', quantity='x', dimension_type='spatial'))
    data_set.set_dimension(1, sid.Dimension(y_dim, name='y', units='m', quantity='y', dimension_type='spatial'))
    data_set.set_dimension(2, sid.Dimension(kx_dim, name='Intensity KX', units='counts',
                                            quantity='Intensity', dimension_type='spectral'))
    data_set.set_dimension(3, sid.Dimension(ky_dim, name='Intensity KY', units='counts',
                                            quantity='Intensity', dimension_type='spectral'))
    data_set.metadata = parms_dict
    return data_set, [kx.ravel(), ky.ravel()]
