#Define the guess and fit functions required to test the SidpyFitterRefactor

import numpy as np
from .fitter_function_utils import gaussian_2d, gaussian_2d_guess, loop_fit_function, generate_guess, generate_deep_guess, SHO_fit_flattened, sho_guess_fn
import tempfile
from pathlib import Path
import unittest
from sidpy.proc.fitter_refactor import SidpyFitterRefactor
import sidpy as sid
import SciFiReaders as sr
from urllib.request import urlretrieve
import logging

log = logging.getLogger(__name__)

class TestSidpyFitterRefactor(unittest.TestCase):
    
    def test_beps_fit(self):
        with tempfile.TemporaryDirectory() as d:
            tmp_path = Path(d)

            #from sidpy.proc.fitter_refactor import SidpyFitterRefactor
            #import SciFiReaders as sr
            #import numpy as np

            from urllib.request import urlretrieve

            url = "https://github.com/pycroscopy/DTMicroscope/raw/refs/heads/main/data/AFM/BEPS_PTO_50x50.h5"
            beps_file = tmp_path / "BEPS_PTO_50x50.h5"
            
            urlretrieve(url, beps_file)
            self.assertTrue(beps_file.exists())

            reader = sr.NSIDReader(str(beps_file))
            data = reader.read()['Channel_000']

            #Crop it for testing, roll it
            data_loop_cropped = data[10:20,10:20,:]

            for ind_x in range(data_loop_cropped.shape[0]):
                for ind_y in range(data_loop_cropped.shape[1]):
                    data_loop_cropped[ind_x, ind_y,: ] = np.roll(np.array(data_loop_cropped[ind_x, ind_y,:]), 16)*1E3

            dc_vec = data._axes[2].values
            dc_vec_rolled = np.roll(dc_vec, 16)
            import sidpy as sid
            data_loop_cropped.set_dimension(2, sid.Dimension(dc_vec_rolled, name = 'DC Offset', quantity = 'Voltage', units = 'Volts', 
                                                        dimension_type = 'spectral'))
            
            fitter = SidpyFitterRefactor(data_loop_cropped, loop_fit_function, generate_guess, ind_dims = (2,))
            fitter.setup_calc()

            log.info("Testing Guess Function in test_beps_fit")
            beps_guess = fitter.do_guess()
            
            #option 1: No Prior Fitting (Better for clean data)
            log.info('Testing Fitting without K-Means in test_beps_fit')
            result_beps = fitter.do_fit(use_kmeans=False, n_clusters=12)
            

            log.info("Testing Fitting with K-Means in test_beps_fit, with covariance and huber loss")
            # Option 2: K-Means Prior Fitting (Better for noisy data)
            result_beps = fitter.do_fit(use_kmeans=True, n_clusters=6, return_cov=True, loss='huber')
            
            #Check to see that the results metadata is beign written correctly
            model_source = result_beps[0].metadata['source_code']['model_function']

            from scipy.special import erf
            context = {'erf': erf}
            #Reload the model, see how the fits shake up
            reloaded_model = fitter.reconstruct_function(model_source, context=context)
            
            vdc = data_loop_cropped._axes[2].values #vdc vector
            #See how the fits shake up
            pix_x = 3
            pix_y = 4

            raw_loop = data_loop_cropped[pix_y, pix_x,:]
            fit_loop = reloaded_model(vdc, *np.array(data_loop_cropped[pix_y, pix_x,:]))
            assert np.isfinite((fit_loop-raw_loop).mean()) #ensure that the fit is valid, and we are reading the fit function correctly

    def test_sho_fit(self):
        with tempfile.TemporaryDirectory() as d:
            tmp_path = Path(d)

            #from sidpy.proc.fitter_refactor import SidpyFitterRefactor
            #import SciFiReaders as sr
            #from urllib.request import urlretrieve

            url = "https://www.dropbox.com/scl/fi/wmablmstf3gw0dokzen6o/PTO_60x60_3rdarea_0003.h5?rlkey=ozr89y9ztggznj2p7fjls0zz2&dl=1"
            SHO_dataset = tmp_path / 'PTO_60x60_3rdarea_0003.h5'
            
            urlretrieve(url, SHO_dataset)
            self.assertTrue(SHO_dataset.exists())

            reader = sr.Usid_reader(str(SHO_dataset))
            data_sho = reader.read()[0] #read the data

            freq_axis = data_sho.labels.index('Frequency (Hz)') #grab the frequency axis
            freq_vec = data_sho._axes[freq_axis].values
            data_sho_cropped = data_sho[:10,:10,:,:,:]
            
            fitter_sho = SidpyFitterRefactor(data_sho_cropped, SHO_fit_flattened, sho_guess_fn, ind_dims=(2,))
            fitter_sho.setup_calc()
            log.info('Testing the Guess function in test_sho_fit')
            guess_results = fitter_sho.do_guess()

            log.info('Testing the Fit function without Kmeans in test_sho_fit')
            parameters_dask = fitter_sho.do_fit(use_kmeans=False)
            fit_results_sho = parameters_dask

            log.info('Testing the Fit function with Kmeans in test_sho_fit')
            parameters_dask = fitter_sho.do_fit(use_kmeans=True, n_clusters=10)
            fit_results_sho = parameters_dask


class TestSidpyFitter2D(unittest.TestCase):

    def setUp(self):
        """
        Create a 3x3x32x32 sidpy.Dataset with synthetic 2D Gaussian data.
        Dimensions: (X_spatial, Y_spatial, kx_spectral, ky_spectral)
        """
        self.n_x, self.n_y = 3, 3    # Spatial
        self.k_x, self.k_y = 32, 32  # Spectral
        
        # Define Axes
        kx_axis = np.linspace(-10, 10, self.k_x)
        ky_axis = np.linspace(-10, 10, self.k_y)
        
        # Container for data
        data_shape = (self.n_x, self.n_y, self.k_x, self.k_y)
        
        # --- FIX: Ground truth should only store params for spatial dims ---
        self.ground_truth = np.zeros((self.n_x, self.n_y, 6)) 
        
        self.raw_data = np.zeros(data_shape)

        # Generate Data
        for i in range(self.n_x):
            for j in range(self.n_y):
                # True parameters for this pixel
                amp = 5.0
                # Shift center slightly based on pixel index (i, j)
                x0 = 2.0 * (i - 1) 
                y0 = 2.0 * (j - 1)
                sigma_x = 2.5
                sigma_y = 1.5
                offset = 1.0
                
                params = [amp, x0, y0, sigma_x, sigma_y, offset]
                
                # Assign to the (3, 3, 6) array
                self.ground_truth[i, j] = params
                
                # Generate noiseless model using the axes
                pixel_data = gaussian_2d([kx_axis, ky_axis], *params)
                
                # Add slight noise
                noise = np.random.normal(0, 0.1, pixel_data.shape)
                self.raw_data[i, j, :, :] = pixel_data + noise

        # Create sidpy Dataset
        self.dataset = sid.Dataset.from_array(self.raw_data, name='Synthetic_2D_Gauss')
        
        # Set Dimensions
        self.dataset.set_dimension(0, sid.Dimension(np.arange(self.n_x), 'x', dimension_type='spatial'))
        self.dataset.set_dimension(1, sid.Dimension(np.arange(self.n_y), 'y', dimension_type='spatial'))
        self.dataset.set_dimension(2, sid.Dimension(kx_axis, 'kx', dimension_type='spectral'))
        self.dataset.set_dimension(3, sid.Dimension(ky_axis, 'ky', dimension_type='spectral'))
        
        self.param_labels = ['Amp', 'x0', 'y0', 'SigX', 'SigY', 'Offset']

    
    def test_2d_fit_execution(self):
        """
        Test the SidpyFitterRefactor on the 4D dataset (2D spectral fits).
        """
        fitter = SidpyFitterRefactor(
            dataset=self.dataset,
            model_function=gaussian_2d,
            guess_function=gaussian_2d_guess,
            ind_dims=(2, 3), # Explicitly stating kx, ky are the fit dims
            num_params=6
        )
        
        # Run Fit with Covariance and Custom Labels
        # Using a simple linear loss for speed/demonstration
        res_params, res_cov = fitter.do_fit(
            use_kmeans=False, 
            fit_parameter_labels=self.param_labels,
            return_cov=True,
            loss='linear'
        )
        
        # --- Assertions ---
        
        # 1. Check Output Types
        self.assertIsInstance(res_params, sid.Dataset, "Params should be a sidpy Dataset")
        self.assertIsInstance(res_cov, sid.Dataset, "Covariance should be a sidpy Dataset")
        
        # 2. Check Shapes
        # Params shape: (3, 3, 6)
        expected_param_shape = (self.n_x, self.n_y, 6)
        self.assertEqual(res_params.shape, expected_param_shape)
        
        # Cov shape: (3, 3, 6, 6)
        expected_cov_shape = (self.n_x, self.n_y, 6, 6)
        self.assertEqual(res_cov.shape, expected_cov_shape)
        
        # 3. Check Parameter Accuracy
        # Pick the center pixel (1, 1) where true x0=0, y0=0
        # True params: [5.0, 0.0, 0.0, 2.5, 1.5, 1.0]
        true_p = self.ground_truth[1, 1]
        fit_p = np.array(res_params[1, 1])
        
        print(f"\nCenter Pixel Truth: {true_p}")
        print(f"Center Pixel Fit:   {fit_p}")
        
        # Allow some tolerance due to noise
        np.testing.assert_allclose(fit_p, true_p, rtol=0.1, atol=0.2, 
                                   err_msg="Fit parameters diverged too much from ground truth")

        # 5. Check Covariance Validity
        # Covariance diagonal (variance) should be positive
        cov_matrix = np.array(res_cov[1, 1])
        diag = np.diag(cov_matrix)
        self.assertTrue(np.all(diag >= 0), "Covariance diagonal elements must be non-negative")

class TestSidpyFitterWithBounds(unittest.TestCase):

    def setUp(self):
        """
        Synthetic 3x3 spatial x 50-point spectral 1D Gaussian dataset.
        Fast, self-contained, no network required.
        """
        self.n_x, self.n_y, self.n_spec = 3, 3, 50
        x_axis = np.linspace(-10, 10, self.n_spec)
        self.true_params = np.array([3.0, 0.0, 2.0, 0.5])  # amp, center, sigma, offset

        raw = np.zeros((self.n_x, self.n_y, self.n_spec))
        for i in range(self.n_x):
            for j in range(self.n_y):
                amp, cen, sig, off = self.true_params
                cen_ij = cen + 0.5 * (i - 1)
                raw[i, j] = (amp * np.exp(-0.5 * ((x_axis - cen_ij) / sig) ** 2)
                             + off
                             + np.random.default_rng(i * 10 + j).normal(0, 0.05, self.n_spec))

        self.dataset = sid.Dataset.from_array(raw, name='Synthetic_1D_Gauss')
        self.dataset.set_dimension(0, sid.Dimension(np.arange(self.n_x), 'x',
                                                    dimension_type='spatial'))
        self.dataset.set_dimension(1, sid.Dimension(np.arange(self.n_y), 'y',
                                                    dimension_type='spatial'))
        self.dataset.set_dimension(2, sid.Dimension(x_axis, 'spectrum',
                                                    dimension_type='spectral'))

        def _gaussian(x, amp, cen, sig, off):
            return amp * np.exp(-0.5 * ((x - cen) / sig) ** 2) + off

        def _gaussian_guess(x, y):
            off = np.percentile(y, 10)
            amp = float(y.max()) - off
            cen = float(x[np.argmax(y)])
            sig = (x[-1] - x[0]) / 6.0
            return [amp, cen, sig, off]

        self.model_func = _gaussian
        self.guess_func = _gaussian_guess

    def _make_fitter(self, lower_bounds=None, upper_bounds=None):
        """Helper: build and setup a fitter with optional bounds."""
        fitter = SidpyFitterRefactor(
            self.dataset, self.model_func, self.guess_func,
            lower_bounds=lower_bounds, upper_bounds=upper_bounds,
        )
        fitter.setup_calc()
        return fitter

    def test_unbounded_unchanged(self):
        """Unbounded fit must return a valid sidpy.Dataset with finite params."""
        result, _ = self._make_fitter().do_fit()
        self.assertIsInstance(result, sid.Dataset)
        self.assertTrue(np.all(np.isfinite(np.array(result))))

    def test_scalar_lower_bound(self):
        """Scalar lower_bounds=0 — all returned params must be >= 0."""
        result, _ = self._make_fitter(lower_bounds=0.0).do_fit()
        params = np.array(result)
        self.assertTrue(np.all(params >= -1e-6),
                        msg=f"Some params violated lower_bound=0: min={params.min()}")

    def test_scalar_upper_bound(self):
        """Scalar upper_bounds=1e6 — all returned params must be <= 1e6."""
        upper = 1e6
        result, _ = self._make_fitter(upper_bounds=upper).do_fit()
        params = np.array(result)
        self.assertTrue(np.all(params <= upper + 1e-6),
                        msg=f"Some params violated upper_bound={upper}: max={params.max()}")

    def test_per_param_bounds_respected(self):
        """Per-parameter array bounds — each param stays within its own [lb, ub]."""
        n = self._make_fitter().num_params
        lb = np.zeros(n)
        ub = np.full(n, 1e6)
        result, _ = self._make_fitter(lower_bounds=lb, upper_bounds=ub).do_fit()
        params = np.array(result)
        for i in range(n):
            p = params[..., i]
            self.assertTrue(np.all(p >= lb[i] - 1e-6),
                            msg=f"Param {i} violated lower bound {lb[i]}: min={p.min()}")
            self.assertTrue(np.all(p <= ub[i] + 1e-6),
                            msg=f"Param {i} violated upper bound {ub[i]}: max={p.max()}")

    def test_guess_outside_bounds_no_crash(self):
        """Guess outside bounds must be clipped silently, not crash."""
        n = self._make_fitter().num_params
        fitter = self._make_fitter(lower_bounds=np.full(n, -1e-10),
                                   upper_bounds=np.full(n,  1e-10))
        try:
            fitter.do_fit()
        except Exception as e:
            self.fail(f"do_fit raised unexpectedly with out-of-bounds guess: {e}")

    def test_bounds_length_mismatch_raises(self):
        """Bound array with wrong length must raise ValueError."""
        n = self._make_fitter().num_params
        fitter = self._make_fitter(lower_bounds=np.zeros(n + 3))
        with self.assertRaises(ValueError):
            fitter.do_fit()

    def test_lb_greater_than_ub_raises(self):
        """lower_bounds > upper_bounds must raise ValueError."""
        n = self._make_fitter().num_params
        fitter = self._make_fitter(lower_bounds=np.full(n, 10.0),
                                   upper_bounds=np.full(n,  1.0))
        with self.assertRaises(ValueError):
            fitter.do_fit()

    def test_bounds_stored_in_metadata(self):
        """Bounds must appear correctly in result metadata."""
        n = self._make_fitter().num_params
        lb = list(np.zeros(n))
        ub = list(np.ones(n) * 1e6)
        result, _ = self._make_fitter(lower_bounds=lb, upper_bounds=ub).do_fit()
        meta = result.metadata["fit_parameters"]
        self.assertIn("lower_bounds", meta)
        self.assertIn("upper_bounds", meta)
        self.assertEqual(meta["lower_bounds"], lb)
        self.assertEqual(meta["upper_bounds"], ub)

    def test_none_bounds_metadata_is_none(self):
        """When no bounds are passed, metadata entries must be None."""
        result, _ = self._make_fitter().do_fit()
        meta = result.metadata["fit_parameters"]
        self.assertIsNone(meta.get("lower_bounds"))
        self.assertIsNone(meta.get("upper_bounds"))

    def test_bounds_with_nonlinear_loss(self):
        """Bounds + non-linear loss must not raise (both require method='trf')."""
        fitter = self._make_fitter(lower_bounds=0.0)
        try:
            result, _ = fitter.do_fit(loss='soft_l1')
        except Exception as e:
            self.fail(f"do_fit raised with bounds + non-linear loss: {e}")
        self.assertIsInstance(result, sid.Dataset)

    def test_bounds_with_return_cov(self):
        """Bounds must be compatible with return_cov=True; covariance shape check."""
        n = self._make_fitter().num_params
        params, cov = self._make_fitter(lower_bounds=0.0).do_fit(return_cov=True)
        self.assertEqual(cov.shape[-2:], (n, n),
                         msg=f"Covariance shape {cov.shape} does not end in ({n},{n})")