import numpy as np
import dask.array as da
from scipy.optimize import least_squares
#from sklearn.cluster import KMeans
# Try importing dask_ml, warn if missing
try:
    from dask_ml.cluster import KMeans as DaskKMeans
except ImportError:
    raise ImportError("The 'dask-ml' library is required for scalable K-Means. Install via 'pip install dask-ml'.")
import inspect
import sidpy as sid

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

    def __init__(self, dataset, model_function, guess_function, ind_dims=None):
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
            The indices of the dimensions to fit over. Default is whatever are the spectral dimensions
        """
        import sidpy
        if not isinstance(dataset, sidpy.Dataset):
            raise TypeError("Dataset must be a sidpy.Dataset object.")
        
        self.dataset = dataset
        self.dask_data = dataset
        self.model_func = model_function
        self.guess_func = guess_function
        
        self.ndim = self.dataset.ndim
        self.spectral_dims = tuple(self.dataset.get_spectral_dims())

        self.ind_dims = tuple(self.spectral_dims ) if ind_dims is None else ind_dims
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
        """Extracts source code and splits into lines for readable metadata."""
        try:
            raw_source = inspect.getsource(func)
            return raw_source.splitlines()  # Returns list of strings
        except (TypeError, OSError):
            return ["Source code not available (function might be defined in a shell or compiled)."]

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

    def _fit_logic(self, y_vec, x_in, initial_guess, loss='linear', f_scale=1.0):
        """
        Core optimization logic for a single pixel.
        Can handle robust fitting via least_squares method as well.
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
            # Robust args passed here
            res = least_squares(residuals, initial_guess, args=(x_in, y_input),
                                loss=loss, f_scale=f_scale)
        else:
            def residuals(p, x, y):
                fit = np.ravel(self.model_func(x, *p))
                return y - fit
            # Robust args passed here
            res = least_squares(residuals, initial_guess, args=(x_in, y_vec),
                                loss=loss, f_scale=f_scale)
        return res.x

    def do_kmeans_guess(self, n_clusters=10):
        """
        Performs K-Means clustering to find representative spectra for prior fitting.
        We use Dask-ML Kmeans to do this in a scalable fashion.

        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters to use for K-Means. Default is 10.

        Returns
        -------
        dask.array.Array
            A dask array containing the initial guesses for every pixel.
        """
        print(f"Starting Dask K-Means Guess with {n_clusters} clusters...")
        
        # 1. Prepare Data as Flat Dask Array (Pixels, Features)
        # We use the internal self.dask_data which might be rechunked already
        # Move spectral dims to the end
        data_move = da.moveaxis(self.dask_data, self.ind_dims, range(-len(self.ind_dims), 0))
        
        n_spectral = np.prod([self.dataset.shape[d] for d in self.ind_dims])
        # Reshape to (Pixels, Spectrum)
        flat_data = data_move.reshape((-1, int(n_spectral)))
        
        # 2. Normalize (Lazy Dask Operations)
        clustering_data = da.absolute(flat_data) if self.is_complex else flat_data
        
        # Compute min/max per pixel (axis 1) keeping dims for broadcasting
        d_min = clustering_data.min(axis=1, keepdims=True)
        d_max = clustering_data.max(axis=1, keepdims=True)
        denom = (d_max - d_min)
        # Avoid divide by zero
        denom = da.where(denom == 0, 1.0, denom)
        
        norm_data = (clustering_data - d_min) / denom
       
        #print("Norm data shape is {}".format(norm_data.shape))
        # 3. Fit K-Means using Dask-ML
        # init='k-means||' is the scalable version of k-means++
        km = DaskKMeans(n_clusters=n_clusters, init='k-means||', random_state=42)
        km.fit(norm_data.squeeze()) # This triggers computation for centroids
        
        labels = km.labels_ # This is a dask array of shape (Pixels,)

        print("Calculating cluster means and fitting priors...")

        # 4. Compute Mean Spectra for each cluster (Original Data)
        # We need the mean of the *original* flat_data based on the labels.
        # We construct a list of lazy mean computations and compute them in one pass.
        lazy_means = []
        for i in range(n_clusters):
            mask = (labels == i)
            # Dask boolean indexing -> Mean. 
            # Note: Boolean indexing creates unknown chunk sizes, but mean collapses them.
            # We add a check for empty clusters to avoid NaNs.
            cluster_mean = flat_data[mask].mean(axis=0)
            lazy_means.append(cluster_mean.squeeze())
        
        # Compute all means simultaneously to read data once
        computed_means = da.compute(*lazy_means)
        
        # 5. Fit the priors (runs locally as n_clusters is small)
        priors_per_cluster = np.zeros((n_clusters, self.num_params))
        
        for i, mean_spec in enumerate(computed_means):
            # Handle empty clusters (NaNs)
            if np.any(np.isnan(mean_spec)):
                priors_per_cluster[i] = np.zeros(self.num_params) 
                continue

            init_p = self.guess_func(self.x_axis, mean_spec)
            priors_per_cluster[i] = self._fit_logic(mean_spec, self.x_axis, init_p)
        
        # 6. Map priors back to full spatial shape
        # labels is a dask array, we can index the numpy priors array with it
        # We need to map_blocks this look-up because 'priors_per_cluster' is numpy
        # and 'labels' is dask.
        
        def map_priors(label_block, priors_lookup):
            return priors_lookup[label_block]

        full_prior_flat = labels.map_blocks(
            map_priors, 
            priors_lookup=priors_per_cluster, 
            dtype=priors_per_cluster.dtype,
            chunks=labels.chunks + (self.num_params,),
            new_axis=1 # Adding the parameter axis
        )

        # Reshape back to (Spatial..., Params)
        spatial_shape = [self.dataset.shape[d] for d in self.spat_dims]
        full_prior_map = full_prior_flat.reshape(tuple(spatial_shape) + (self.num_params,))
        
        return full_prior_map

    @staticmethod
    def reconstruct_function(source_code_input, context=None):
        """
        Reconstructs a python function from source code stored in metadata.
        Robustly handles lists, strings, and indentation issues.
        """
        import textwrap
        import numpy as np
        # 1. STANDARDIZE INPUT TO STRING
        # The input might be a list of strings (new format), a single string (old format), 
        # or a numpy array of strings (from some HDF5 loaders).
        
        if isinstance(source_code_input, str):
            # It's already a string
            source_code = source_code_input
        elif isinstance(source_code_input, (list, tuple, np.ndarray)):
            # It's a sequence: join it into a single block
            # map(str, ...) handles cases where data might be numpy objects
            source_code = "\n".join(map(str, source_code_input))
        else:
            raise TypeError(f"Source code must be a string or list of strings, not {type(source_code_input)}")

        # 2. FIX INDENTATION
        # textwrap.dedent ONLY works on strings, which is why we joined above.
        # This removes the common leading whitespace (e.g. if defined inside a class).
        source_code = textwrap.dedent(source_code)

        # 3. SETUP CONTEXT (Imports)
        local_scope = {}
        global_scope = context if context is not None else {}
        
        # Ensure numpy is available by default as 'np'
        if 'np' not in global_scope:
            global_scope['np'] = np

        # 4. EXECUTE
        try:
            exec(source_code, global_scope, local_scope)
        except Exception as e:
            print(f"--- SOURCE CODE ERROR ---")
            print(source_code)
            print(f"-------------------------")
            raise RuntimeError(f"Failed to execute source code. Error: {e}")

        # 5. EXTRACT CALLABLE
        # Find the function definition in the local scope
        callables = {k: v for k, v in local_scope.items() 
                     if callable(v) and k != '__builtins__'}
        
        if not callables:
            raise ValueError("No function was defined in the provided source code.")
        
        # Return the last defined function (most likely the target)
        return list(callables.values())[-1]
        
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

        self.guess_result =  self.dask_data.map_blocks(
            guess_worker, self.x_axis, self.ind_dims, self.num_params,
            dtype=np.float32, drop_axis=self.ind_dims, new_axis=[self.ndim]
        )

        return self.guess_result.compute() #this is still a dask array only, we won't return sidpy arrays at this intermediate stage.

    def do_fit(self, guesses=None, use_kmeans=False, n_clusters=10, 
               fit_parameter_labels=None, loss='linear', f_scale=1.0):
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
        fit_parameter_labels : list of str, optional
            List of string labels for the fit parameters (e.g. ['Amp', 'Phase']).
        loss : str, optional
            Loss function for least_squares (e.g., 'linear', 'soft_l1', 'huber', 'cauchy', 'arctan').
        f_scale : float, optional
             Value of soft margin between inlier and outlier residuals.

        Returns
        -------
        dask.array.Array
            Dask array containing the optimized fit parameters.
        """
        # Store labels for transform step
        self.fit_parameter_labels = fit_parameter_labels

        # Update metadata for the current run
        self.metadata["fit_parameters"]["use_kmeans"] = use_kmeans
        self.metadata["fit_parameters"]["n_clusters"] = n_clusters if use_kmeans else None
        self.metadata["fit_parameters"]["loss"] = loss
        self.metadata["fit_parameters"]["f_scale"] = f_scale

        if guesses is None:
            guesses = self.do_kmeans_guess(n_clusters) if use_kmeans else self.do_guess()

        # Capture loss/f_scale in the closure
        def fit_worker(data_block, guess_block, x_in, ind_dims, num_params):
            data_block, guess_block = np.asarray(data_block), np.asarray(guess_block)
            flat_data, spat_shape = self._prepare_block(data_block, ind_dims)
            flat_guess = guess_block.reshape(-1, guess_block.shape[-1])
            
            out_flat = np.zeros((flat_data.shape[0], num_params))
            for i in range(flat_data.shape[0]):
                if flat_data[i].size == 0: continue
                # Pass robust args here
                out_flat[i] = self._fit_logic(flat_data[i], x_in, flat_guess[i],
                                              loss=loss, f_scale=f_scale)
            return out_flat.reshape(spat_shape + (num_params,))

        data_ind = tuple(range(self.ndim))
        guess_ind = tuple(self.spat_dims + [self.ndim])

        self.fit_result =  da.blockwise(
            fit_worker, guess_ind,
            self.dask_data, data_ind,
            guesses, guess_ind,
            self.x_axis, None,
            self.ind_dims, None,
            self.num_params, None,
            dtype=np.float32, align_arrays=True, concatenate=True
        )

        computed_result = self.fit_result.compute()
        self.sidpy_result = self.transform_to_sidpy(computed_result)

        return self.sidpy_result

    def transform_to_sidpy(self, fit_dask_array):
        """
        Convert the dask array of fit results into a sidpy.Dataset.

        The returned dataset will have the original spatial dimensions (those
        that were not used as independent/spectral dims for fitting) and a
        final spectral-like dimension that indexes the fit parameters. The
        fit-parameter axis will be given a generic name and marked as a
        spectral dimension.

        Parameters
        ----------
        fit_dask_array : dask.array.Array
            Dask array produced by the fitting routine. Expected shape is
            spatial_shape + (num_params,)

        Returns
        -------
        sidpy.Dataset
            Dataset view of the fit results. Also stored on the instance as
            `self.fit_dataset`.
        """

        # Create a sidpy Dataset from the dask array (preserves lazy compute)
        sid_dset = sid.Dataset.from_array(fit_dask_array, title=f"{self.dataset.title}_fit",
                                       chunks='auto')

        # Recreate spatial dimensions in the same order as self.spat_dims
        for i, orig_dim in enumerate(self.spat_dims):
            try:
                orig_axis = self.dataset._axes[orig_dim].copy()
            except Exception:
                # Fallback: create a generic axis of appropriate length
                orig_axis = sid.Dimension(np.arange(self.dataset.shape[orig_dim]),
                                      name=f"dim_{orig_dim}", dimension_type='spatial')
            sid_dset.set_dimension(i, orig_axis)

        # Add fit-parameter axis as a spectral-like dimension
        # Check for custom labels provided in do_fit
        custom_labels = getattr(self, 'fit_parameter_labels', None)
        
        # Determine values for the parameter axis
        if custom_labels is not None and len(custom_labels) == fit_dask_array.shape[-1]:
            param_values = np.array(custom_labels)
            quantity = 'Label'
        else:
            param_values = np.arange(self.num_params if self.num_params is not None else fit_dask_array.shape[-1])
            quantity = 'index'

        param_axis = sid.Dimension(param_values, name='fit_parameters', quantity=quantity,
                               units='a.u.', dimension_type='spectral')
        sid_dset.set_dimension(len(self.spat_dims), param_axis)

        # Attach provenance and metadata about the fit
        sid_dset.metadata = dict(self.metadata).copy()
        # link back to parent
        sid_dset.provenance = {'sidpy': {'generated_from': self.dataset.title}}
        sid_dset.add_provenance('sidpy.proc.SidpyFitterRefactor', 'transform_to_sidpy', version=1,
                          linked_data={'parent': self.dataset.title})
        sid_dset.data_type = self.dataset.data_type

        return sid_dset
    

    
    
