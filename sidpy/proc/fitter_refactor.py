import numpy as np
import dask.array as da
from scipy.optimize import least_squares
from sklearn.cluster import KMeans
import inspect

class SidpyFitterRefactor:
    """
    A parallelized fitter for sidpy.Datasets that supports K-Means-based 
    initial guesses for improved convergence on large datasets.
    
    Attributes
    ----------
    dataset : sidpy.Dataset
        The original sidpy dataset containing data and metadata.
    dask_data : dask.array.Array
        The underlying dask array used for parallel computation.
    model_func : callable
        The function to fit. Expected signature: f(x_axis, *params).
    guess_func : callable
        The function to generate initial guesses. Expected signature: f(x_axis, y_data).
    metadata : dict
        A dictionary containing fit parameters, model source code, and configuration.
    """

    def __init__(self, dataset, model_function, guess_function, ind_dims=(2,)):
        """
        Initializes the SidpyFitterKMeans.

        Inputs
        ----------
        dataset : sidpy.Dataset
            Dataset to be fitted.
        model_function : callable
            The model function to use for fitting.
        guess_function : callable
            The function to generate initial parameters for the model.
        ind_dims : int or tuple of int, optional
            The indices of the dimensions to fit over. Default is (2,).
            #TODO: Change the default to be over the existing spectral dimension
        """
        import sidpy
        if not isinstance(dataset, sidpy.Dataset):
            raise TypeError("Dataset must be a sidpy.Dataset object.")
        
        self.dataset = dataset
        self.dask_data = dataset
        self.model_func = model_function
        self.guess_func = guess_function
        
        self.ndim = self.dataset.ndim
        self.ind_dims = tuple(ind_dims) if isinstance(ind_dims, int) else ind_dims
        self.spat_dims = [d for d in range(self.ndim) if d not in self.ind_dims]
        
        # Standardize x_axis (coordinate values)
        self.x_axis = np.array([self.dataset._axes[d].values for d in self.ind_dims]).squeeze()
        
        self.is_complex = np.iscomplexobj(self.dataset)
        self.num_params = None

        # --- Reproducibility Metadata ---
        self.metadata = {
            "fit_parameters": {
                "ind_dims": self.ind_dims,
                "is_complex": self.is_complex,
                "use_kmeans": False,  # Updated during do_fit
                "n_clusters": None     # Updated during do_fit
            },
            "source_code": {
                "model_function": self._get_source(model_function),
                "guess_function": self._get_source(guess_function)
            },
            "dataset_info": {
                "original_name": self.dataset.name,
                "original_shape": self.dataset.shape
            }
        }

    def _get_source(self, func):
        """Extracts source code from a function for metadata storage."""
        try:
            return inspect.getsource(func)
        except (TypeError, OSError):
            return "Source code not available (function might be defined in a shell or compiled)."

    def setup_calc(self, chunks='auto'):
        """
        Prepares the calculation by rechunking and determining the parameter count.

        Parameters
        ----------
        chunks : str or tuple, optional
            The chunk size for the dask array. Default is 'auto'.
        """
        if chunks:
            self.dask_data = self.dask_data.rechunk(chunks)
        
        s_slice = [0] * self.ndim
        for d in self.ind_dims: 
            s_slice[d] = slice(None)
        
        sample_y = np.array(self.dask_data[tuple(s_slice)]).ravel()
        sample_guess = np.asarray(self.guess_func(self.x_axis, sample_y)).ravel()
        self.num_params = len(sample_guess)
        
        self.metadata["num_params"] = self.num_params
        print(f"Setup Complete. Params: {self.num_params} | Spatial Dims: {self.spat_dims}")

    def _prepare_block(self, block, ind_dims):
        """Internal helper to reshape dask blocks into (Pixels, Spectrum)."""
        n_ind = len(ind_dims)
        dest = tuple(range(block.ndim - n_ind, block.ndim))
        data = np.moveaxis(block, ind_dims, dest)
        spatial_shape = data.shape[:-n_ind]
        flat_data = data.reshape(-1, np.prod(data.shape[-n_ind:]))
        return flat_data, spatial_shape

    def _fit_logic(self, y_vec, x_in, initial_guess):
        """
        Core optimization logic for a single pixel.
        """
        y_vec = np.squeeze(np.asarray(y_vec))
        initial_guess = np.asarray(initial_guess).ravel()
        
        if self.is_complex:
            y_input = np.hstack([y_vec.real, y_vec.imag])
            def residuals(p, x, y_s):
                fit = np.squeeze(self.model_func(x, *p))
                if fit.size != y_s.size:
                    fit = np.hstack([fit.real, fit.imag])
                return y_s - fit
            res = least_squares(residuals, initial_guess, args=(x_in, y_input))
        else:
            def residuals(p, x, y):
                fit = np.ravel(self.model_func(x, *p))
                return y - fit
            res = least_squares(residuals, initial_guess, args=(x_in, y_vec))
        return res.x

    def do_kmeans_guess(self, n_clusters=10):
        """
        Performs K-Means clustering to find representative spectra for prior fitting.

        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters to use for K-Means. Default is 10.

        Returns
        -------
        dask.array.Array
            A dask array containing the initial guesses for every pixel.
        """
        print(f"Starting K-Means Guess with {n_clusters} clusters...")
        
        # Cast to base Dask Array to bypass sidpy overrides
        pure_dask = da.Array(self.dataset.dask, self.dataset.name, 
                             self.dataset.chunks, self.dataset.dtype)
        
        n_spectral = np.prod([self.dataset.shape[d] for d in self.ind_dims])
        total_pixels = self.dataset.size // n_spectral
        
        data_move = da.moveaxis(pure_dask, self.ind_dims, -1)
        flat_data = data_move.reshape((int(total_pixels), int(n_spectral))).compute()
        
        clustering_data = np.abs(flat_data) if self.is_complex else flat_data
        denom = (clustering_data.max(axis=1, keepdims=True) - 
                 clustering_data.min(axis=1, keepdims=True) + 1e-12)
        norm_data = (clustering_data - clustering_data.min(axis=1, keepdims=True)) / denom

        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(norm_data)

        print("Fitting cluster means...")
        priors_per_cluster = np.zeros((n_clusters, self.num_params))
        for i in range(n_clusters):
            mask = (labels == i)
            if not np.any(mask): continue
            mean_spec = flat_data[mask].mean(axis=0)
            init_p = self.guess_func(self.x_axis, mean_spec)
            priors_per_cluster[i] = self._fit_logic(mean_spec, self.x_axis, init_p)
        
        full_prior_flat = priors_per_cluster[labels]
        spatial_shape = [self.dataset.shape[d] for d in self.spat_dims]
        full_prior_map = full_prior_flat.reshape(spatial_shape + [self.num_params])
        
        return da.from_array(full_prior_map, chunks='auto')

    def do_guess(self):
        """Parallelized guess logic across all pixels."""
        def guess_worker(block, x_in, ind_dims, num_params):
            block = np.asarray(block)
            flat_data, spat_shape = self._prepare_block(block, ind_dims)
            out_flat = np.zeros((flat_data.shape[0], num_params))
            for i in range(flat_data.shape[0]):
                res = self.guess_func(x_in, flat_data[i])
                out_flat[i] = np.asarray(res).ravel()
            return out_flat.reshape(spat_shape + (num_params,))

        return self.dask_data.map_blocks(
            guess_worker, self.x_axis, self.ind_dims, self.num_params,
            dtype=np.float32, drop_axis=self.ind_dims, new_axis=[self.ndim]
        )

    def do_fit(self, guesses=None, use_kmeans=False, n_clusters=10):
        """
        Executes the parallel fit across the dataset.

        Parameters
        ----------
        guesses : dask.array.Array, optional
            Initial guesses. If None, generated automatically.
        use_kmeans : bool, optional
            Whether to use K-means priors. Default is False.
        n_clusters : int, optional
            Number of clusters if use_kmeans is True. Default is 10.

        Returns
        -------
        dask.array.Array
            Dask array containing the optimized fit parameters.
        """
        # Update metadata for the current run
        self.metadata["fit_parameters"]["use_kmeans"] = use_kmeans
        self.metadata["fit_parameters"]["n_clusters"] = n_clusters if use_kmeans else None

        if guesses is None:
            guesses = self.do_kmeans_guess(n_clusters) if use_kmeans else self.do_guess()

        def fit_worker(data_block, guess_block, x_in, ind_dims, num_params):
            data_block, guess_block = np.asarray(data_block), np.asarray(guess_block)
            flat_data, spat_shape = self._prepare_block(data_block, ind_dims)
            flat_guess = guess_block.reshape(-1, guess_block.shape[-1])
            
            out_flat = np.zeros((flat_data.shape[0], num_params))
            for i in range(flat_data.shape[0]):
                if flat_data[i].size == 0: continue
                out_flat[i] = self._fit_logic(flat_data[i], x_in, flat_guess[i])
            return out_flat.reshape(spat_shape + (num_params,))

        data_ind = tuple(range(self.ndim))
        guess_ind = tuple(self.spat_dims + [self.ndim])

        return da.blockwise(
            fit_worker, guess_ind,
            self.dask_data, data_ind,
            guesses, guess_ind,
            self.x_axis, None,
            self.ind_dims, None,
            self.num_params, None,
            dtype=np.float32, align_arrays=True, concatenate=True
        )