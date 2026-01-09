#Define the guess and fit functions required to test the SidpyFitterRefactor

import numpy as np
from .fitter_function_utils import gaussian_2d, gaussian_2d_guess, loop_fit_function, generate_guess, SHO_fit_flattened, sho_guess_fn
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
            beps_guess = fitter.do_guess().compute()
            
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
            guess_results = fitter_sho.do_guess().compute()

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
            loss='linear',
            return_metrics=False
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

    def test_cov_mode_diag_returns_param_length_per_pixel(self):
        """
        cov_mode='diag' should return params + diag(cov) where diag has length num_params.
        For 2D Gaussian case in this file: num_params=6.
        """
        # Create a small synthetic 4D dataset: (X, Y, kx, ky)
        nx, ny = 4, 3
        nkx, nky = 16, 16

        x_vec = np.linspace(-10, 10, nkx)
        y_vec = np.linspace(-10, 10, nky)
        X, Y = np.meshgrid(x_vec, y_vec, indexing='ij')

        # Parameters: amp, x0, y0, sx, sy, offset
        true_p = np.array([1.5, 1.0, -2.0, 2.5, 1.5, 0.1], dtype=float)

        # Use gaussian_2d from fitter_function_utils.py if it exists in your test file imports
        # gaussian_2d expects (axes_dims, *params) or (x_in, *params) depending on your setup;
        # Here we follow your existing TestSidpyFitter2D pattern: model_function=gaussian_2d
        data_2d = gaussian_2d((x_vec, y_vec), *true_p)  # returns 2D array (nkx, nky)
        data_4d = np.tile(data_2d, (nx, ny, 1, 1))

        dset = sid.Dataset.from_array(data_4d, title="gaussian_2d_test")
        dset.set_dimension(0, sid.Dimension(np.arange(nx), name='X', dimension_type='spatial'))
        dset.set_dimension(1, sid.Dimension(np.arange(ny), name='Y', dimension_type='spatial'))
        dset.set_dimension(2, sid.Dimension(x_vec, name='kx', dimension_type='spectral'))
        dset.set_dimension(3, sid.Dimension(y_vec, name='ky', dimension_type='spectral'))

        fitter = SidpyFitterRefactor(
            dataset=dset,
            model_function=gaussian_2d,
            guess_function=gaussian_2d_guess,
            ind_dims=(2, 3),
            num_params=6
        )
        fitter.setup_calc()

        res_params, res_cov_diag = fitter.do_fit(
            use_kmeans=False,
            cov_mode='diag',
            loss='linear',
            return_metrics=False
        )

        # Params should be (nx, ny, 6)
        assert res_params.shape == (nx, ny, 6)

        # Diagonal covariance should be (nx, ny, 6)
        assert res_cov_diag.shape == (nx, ny, 6)

        # Basic sanity: diagonal variances should be finite at least somewhere
        arr = np.asarray(res_cov_diag)
        assert np.isfinite(arr).any()


    def test_cov_mode_stderr_nonnegative_and_shape(self):
        """
        cov_mode='stderr' should return params + stderr where stderr has length num_params
        and is non-negative (sqrt of diag(cov), clipped).
        """
        nx, ny = 4, 3
        nkx, nky = 16, 16

        x_vec = np.linspace(-10, 10, nkx)
        y_vec = np.linspace(-10, 10, nky)
        true_p = np.array([2.0, -1.5, 1.0, 3.0, 2.0, 0.2], dtype=float)

        data_2d = gaussian_2d((x_vec, y_vec), *true_p)
        data_4d = np.tile(data_2d, (nx, ny, 1, 1))

        dset = sid.Dataset.from_array(data_4d, title="gaussian_2d_test_stderr")
        dset.set_dimension(0, sid.Dimension(np.arange(nx), name='X', dimension_type='spatial'))
        dset.set_dimension(1, sid.Dimension(np.arange(ny), name='Y', dimension_type='spatial'))
        dset.set_dimension(2, sid.Dimension(x_vec, name='kx', dimension_type='spectral'))
        dset.set_dimension(3, sid.Dimension(y_vec, name='ky', dimension_type='spectral'))

        fitter = SidpyFitterRefactor(
            dataset=dset,
            model_function=gaussian_2d,
            guess_function=gaussian_2d_guess,
            ind_dims=(2, 3),
            num_params=6
        )
        fitter.setup_calc()

        res_params, res_stderr = fitter.do_fit(
            use_kmeans=False,
            cov_mode='stderr',
            loss='linear',
            return_metrics=False
        )

        assert res_params.shape == (nx, ny, 6)
        assert res_stderr.shape == (nx, ny, 6)

        stderr = np.asarray(res_stderr)
        # stderr is sqrt(max(diag, 0.0)) so it should be >= 0 and finite at least somewhere
        assert (stderr >= 0).all() or np.isnan(stderr).all()  # allow NaNs if dof <= 0 everywhere
        assert np.isfinite(stderr).any() or np.isnan(stderr).all()


class TestSidpyFitterMetrics(unittest.TestCase):

    def test_metrics_perfect_fit_r2_one_rmse_zero(self):
       
        nx, ny = 3, 2
        nkx, nky = 20, 20
        x_vec = np.linspace(-10, 10, nkx)
        y_vec = np.linspace(-10, 10, nky)

        true_p = np.array([1.7, 1.2, -1.5, 2.2, 1.6, 0.05], dtype=float)
        data_2d = gaussian_2d((x_vec, y_vec), *true_p)
        data_4d = np.tile(data_2d, (nx, ny, 1, 1))

        dset = sid.Dataset.from_array(data_4d, title="gaussian_2d_metrics_perfect")
        dset.set_dimension(0, sid.Dimension(np.arange(nx), name='X', dimension_type='spatial'))
        dset.set_dimension(1, sid.Dimension(np.arange(ny), name='Y', dimension_type='spatial'))
        dset.set_dimension(2, sid.Dimension(x_vec, name='kx', dimension_type='spectral'))
        dset.set_dimension(3, sid.Dimension(y_vec, name='ky', dimension_type='spectral'))

        fitter = SidpyFitterRefactor(
            dataset=dset,
            model_function=gaussian_2d,
            guess_function=gaussian_2d_guess,
            ind_dims=(2, 3),
            num_params=6
        )
        fitter.setup_calc()

        res_params, res_metrics = fitter.do_fit(use_kmeans=False, loss='linear', return_metrics=True)

        assert res_params.shape == (nx, ny, 6)
        assert res_metrics.shape == (nx, ny, 4)

        m = np.asarray(res_metrics)
        r2 = m[..., 0]
        rmse = m[..., 1]
        assert np.isfinite(m[..., 2]).any()  # AIC
        assert np.isfinite(m[..., 3]).any()  # BIC
        assert np.nanmin(r2) > 0.999
        assert np.nanmax(rmse) < 1e-6

    def test_aic_bic_increase_with_noise_on_same_model(self):
        import sidpy as sid
        rng = np.random.default_rng(0)

        nx, ny = 2, 2
        nkx, nky = 16, 16
        x_vec = np.linspace(-10, 10, nkx)
        y_vec = np.linspace(-10, 10, nky)
        true_p = np.array([1.8, 1.0, -1.0, 2.4, 1.7, 0.05], dtype=float)

        clean2d = gaussian_2d((x_vec, y_vec), *true_p)
        clean = np.tile(clean2d, (nx, ny, 1, 1))
        sigma = 0.02 * np.max(np.abs(clean2d))
        noisy = clean + rng.normal(0.0, sigma, size=clean.shape)

        def mk(arr, title):
            d = sid.Dataset.from_array(arr, title=title)
            d.set_dimension(0, sid.Dimension(np.arange(nx), name='X', dimension_type='spatial'))
            d.set_dimension(1, sid.Dimension(np.arange(ny), name='Y', dimension_type='spatial'))
            d.set_dimension(2, sid.Dimension(x_vec, name='kx', dimension_type='spectral'))
            d.set_dimension(3, sid.Dimension(y_vec, name='ky', dimension_type='spectral'))
            return d

        def run(arr):
            fitter = SidpyFitterRefactor(mk(arr, "tmp"), gaussian_2d, gaussian_2d_guess, ind_dims=(2, 3), num_params=6)
            fitter.setup_calc()
            _, met = fitter.do_fit(use_kmeans=False, loss='linear')
            m = np.asarray(met)
            return np.nanmedian(m[..., 2]), np.nanmedian(m[..., 3])  # AIC, BIC

        aic_clean, bic_clean = run(clean)
        aic_noisy, bic_noisy = run(noisy)

        assert aic_noisy > aic_clean
        assert bic_noisy > bic_clean


    def test_metrics_noisy_fit_lower_r2_higher_rmse_than_clean(self):

        rng = np.random.default_rng(0)

        nx, ny = 3, 2
        nkx, nky = 20, 20
        x_vec = np.linspace(-10, 10, nkx)
        y_vec = np.linspace(-10, 10, nky)

        true_p = np.array([1.7, 1.2, -1.5, 2.2, 1.6, 0.05], dtype=float)
        data_2d = gaussian_2d((x_vec, y_vec), *true_p)
        data_4d_clean = np.tile(data_2d, (nx, ny, 1, 1))

        noise_sigma = 0.01 * np.max(np.abs(data_2d))
        data_4d_noisy = data_4d_clean + rng.normal(0.0, noise_sigma, size=data_4d_clean.shape)

        def make_dataset(arr, title):
            d = sid.Dataset.from_array(arr, title=title)
            d.set_dimension(0, sid.Dimension(np.arange(nx), name='X', dimension_type='spatial'))
            d.set_dimension(1, sid.Dimension(np.arange(ny), name='Y', dimension_type='spatial'))
            d.set_dimension(2, sid.Dimension(x_vec, name='kx', dimension_type='spectral'))
            d.set_dimension(3, sid.Dimension(y_vec, name='ky', dimension_type='spectral'))
            return d

        dset_clean = make_dataset(data_4d_clean, "gaussian_2d_metrics_clean")
        dset_noisy = make_dataset(data_4d_noisy, "gaussian_2d_metrics_noisy")

        def run_metrics(dset):
            fitter = SidpyFitterRefactor(
                dataset=dset,
                model_function=gaussian_2d,
                guess_function=gaussian_2d_guess,
                ind_dims=(2, 3),
                num_params=6
            )
            fitter.setup_calc()
            _, met = fitter.do_fit(use_kmeans=False, loss='linear', return_metrics=True)
            m = np.asarray(met)
            
            return m[..., 0], m[..., 1]  # r2, rmse

        r2_clean, rmse_clean = run_metrics(dset_clean)
        r2_noisy, rmse_noisy = run_metrics(dset_noisy)

        assert np.nanmedian(r2_noisy) < np.nanmedian(r2_clean)
        assert np.nanmedian(rmse_noisy) > np.nanmedian(rmse_clean)

    def test_default_do_fit_returns_params_and_metrics_only(self):
        rng = np.random.default_rng(0)

        nx, ny = 3, 2
        nkx, nky = 20, 20
        x_vec = np.linspace(-10, 10, nkx)
        y_vec = np.linspace(-10, 10, nky)

        true_p = np.array([1.7, 1.2, -1.5, 2.2, 1.6, 0.05], dtype=float)
        data_2d = gaussian_2d((x_vec, y_vec), *true_p)
        data_4d_clean = np.tile(data_2d, (nx, ny, 1, 1))

        noise_sigma = 0.01 * np.max(np.abs(data_2d))
        data_4d_noisy = data_4d_clean + rng.normal(0.0, noise_sigma, size=data_4d_clean.shape)

        def make_dataset(arr, title):
            d = sid.Dataset.from_array(arr, title=title)
            d.set_dimension(0, sid.Dimension(np.arange(nx), name='X', dimension_type='spatial'))
            d.set_dimension(1, sid.Dimension(np.arange(ny), name='Y', dimension_type='spatial'))
            d.set_dimension(2, sid.Dimension(x_vec, name='kx', dimension_type='spectral'))
            d.set_dimension(3, sid.Dimension(y_vec, name='ky', dimension_type='spectral'))
            return d

        dset_noisy = make_dataset(data_4d_noisy, "gaussian_2d_metrics_noisy")
        fitter = SidpyFitterRefactor(
                dataset=dset_noisy,
                model_function=gaussian_2d,
                guess_function=gaussian_2d_guess,
                ind_dims=(2, 3),
                num_params=6
            )
        fitter.setup_calc()
        params, metrics = fitter.do_fit()
        assert metrics.shape[-1] == 4

       



    
        

