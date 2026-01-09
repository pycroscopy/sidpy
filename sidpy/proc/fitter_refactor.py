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

    def __init__(self, dataset, model_function, guess_function, ind_dims=None, num_params=None):
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
        num_params: int, optional but required in case of 2D or higher fitting
            The number of parameters the fitting function expects.
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
        self.num_params = num_params

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

    def _fit_logic(self, y_vec, x_in, initial_guess, loss='linear', f_scale=1.0,
               return_cov=False, cov_mode=None, return_metrics=False):
        """
        Core optimization logic for a single pixel.

        Parameters
        ----------
        return_cov : bool
            Backward-compatible flag. If True and cov_mode is None -> cov_mode='full'.
        cov_mode : {None, 'full', 'diag', 'stderr'}
            None: return parameters only.
            'full': return parameters + flattened covariance matrix (N*N)
            'diag': return parameters + diagonal of covariance (N)
            'stderr': return parameters + sqrt(diag(cov)) (N)
        return_metrics : bool
            If True, append [R^2, RMSE] to the output.
            Metrics are computed on the same y_input space used in optimization
            (real+imag concatenation for complex data).
        """
        y_vec = np.squeeze(np.asarray(y_vec))
        initial_guess = np.asarray(initial_guess).ravel()

        # Resolve cov_mode from legacy flag
        if cov_mode is None and return_cov:
            cov_mode = 'full'

        # Prepare data for least_squares (handle complex)
        if self.is_complex:
            y_input = np.hstack([y_vec.real, y_vec.imag])

            def residuals(p, x, y_s):
                fit = np.squeeze(self.model_func(x, *p))
                if fit.size != y_s.size:
                    fit = np.hstack([fit.real, fit.imag])
                return y_s - fit
        else:
            y_input = y_vec

            def residuals(p, x, y):
                fit = np.ravel(self.model_func(x, *p))
                return y - fit

        # Guard against NaN/inf in data
        if not np.all(np.isfinite(y_input)):
            params = np.full(initial_guess.size, np.nan, dtype=np.float64)
            out_parts = [params]

            if cov_mode is not None:
                if cov_mode == 'full':
                    out_parts.append(np.full(initial_guess.size ** 2, np.nan, dtype=np.float64))
                elif cov_mode in ('diag', 'stderr'):
                    out_parts.append(np.full(initial_guess.size, np.nan, dtype=np.float64))
                else:
                    raise ValueError(f"Unknown cov_mode={cov_mode!r}.")

            if return_metrics:
                out_parts.append(np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float64))

            return np.hstack(out_parts) if len(out_parts) > 1 else params

        # Run Fit
        res = least_squares(
            residuals,
            initial_guess,
            args=(x_in, y_input),
            loss=loss,
            f_scale=f_scale
        )

        params = np.asarray(res.x, dtype=np.float64)
        out_parts = [params]

        # --- Metrics (default) appending ---
        if return_metrics:
            n = int(y_input.size)
            k = int(params.size)
            sse = float(np.sum(res.fun ** 2))
            rmse = float(np.sqrt(sse / y_input.size)) if y_input.size > 0 else np.nan
            y_mean = float(np.mean(y_input))
            sst = float(np.sum((y_input - y_mean) ** 2))
            if sst == 0.0:
                r2 = 1.0 if sse == 0.0 else np.nan
            else:
                r2 = 1.0 - (sse / sst)

            # AIC/BIC (up to an additive constant)
            if n > 0 and sse > 0.0:
                ll_term = n * np.log(sse / n)
                aic = float(ll_term + 2.0 * k)
                bic = float(ll_term + k * np.log(n))
            else:
                aic = np.nan
                bic = np.nan
                
            # Order: [r2, rmse, aic, bic]
            out_parts.append(np.array([r2, rmse, aic, bic], dtype=np.float64))

        # --- Covariance (optional) appending ---
        if cov_mode is not None:
            J = res.jac
            H = J.T @ J
            cov_unscaled = np.linalg.pinv(H)

            dof = y_input.size - params.size
            if dof > 0:
                var_residuals = np.sum(res.fun ** 2) / dof
            else:
                var_residuals = np.nan

            cov_matrix = cov_unscaled * var_residuals

            if cov_mode == 'full':
                cov_payload = np.asarray(cov_matrix, dtype=np.float64).ravel()
            elif cov_mode == 'diag':
                cov_payload = np.diag(cov_matrix).astype(np.float64, copy=False)
            elif cov_mode == 'stderr':
                diag = np.diag(cov_matrix).astype(np.float64, copy=False)
                cov_payload = np.sqrt(np.maximum(diag, 0.0))
            else:
                raise ValueError(f"Unknown cov_mode={cov_mode!r}. Use None, 'full', 'diag', or 'stderr'.")

            out_parts.append(cov_payload)

       
        return np.hstack(out_parts) if len(out_parts) > 1 else params


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
        """Parallelized guess logic across all pixels.

        Returns
        -------
        dask.array.Array
            Lazy Dask array of shape (spatial..., num_params). Call .compute() in user code
            if you want a NumPy array.
        """

        def guess_worker(y_block, x_in, ind_dims, num_params):
            y_block = np.asarray(y_block)

            # Ensure ind_dims is a tuple of ints
            ind_dims = tuple(ind_dims) if ind_dims is not None else tuple()
            n_ind = len(ind_dims)

            # If no ind dims, treat each element as its own "spectrum"
            if n_ind == 0:
                flat_data = y_block.reshape(-1, 1)
                spat_shape = y_block.shape
            else:
                # Move independent (spectral) axes to the end, preserving order
                all_axes = list(range(y_block.ndim))
                spat_axes = [ax for ax in all_axes if ax not in ind_dims]
                perm = spat_axes + list(ind_dims)

                y_perm = np.transpose(y_block, axes=perm)

                spec_shape = y_perm.shape[-n_ind:]          # spectral shape (may be >1D)
                spec_len = int(np.prod(spec_shape))         # flattened spectral length
                spat_shape = y_perm.shape[:-n_ind]          # spatial block shape

                flat_data = y_perm.reshape(-1, spec_len)    # (n_pixels_in_block, spec_len)

            out_flat = np.zeros((flat_data.shape[0], num_params), dtype=np.float64)

            for i in range(flat_data.shape[0]):
                # guess_func contract: guess_func(x_axis, y_vec)
                guess = self.guess_func(x_in, flat_data[i])
                out_flat[i, :] = np.asarray(guess, dtype=np.float64).ravel()

            return out_flat.reshape(spat_shape + (num_params,))

        self.guess_result = self.dask_data.map_blocks(
            guess_worker,
            self.x_axis,
            self.ind_dims,
            self.num_params,
            dtype=np.float64,
            drop_axis=self.ind_dims,
            new_axis=[self.ndim]
        )

        return self.guess_result


    def do_fit(self, guesses=None, use_kmeans=False, n_clusters=10,
           fit_parameter_labels=None, loss='linear', f_scale=1.0,
           return_cov=False, cov_mode=None, return_metrics=True):

        """
        Executes the parallel fit.

        New:
        ----
        cov_mode : {None, 'full', 'diag', 'stderr'}
            - None: return parameters only (default)
            - 'full': return full covariance matrix (N*N) per pixel
            - 'diag': return diag(cov) (N) per pixel
            - 'stderr': return sqrt(diag(cov)) (N) per pixel
            - return_metrics: bool
                If True, append [R^2, RMSE] to the output.

        Backward compatible:
        --------------------
        return_cov=True with cov_mode=None -> cov_mode='full'
        """
        self.fit_parameter_labels = fit_parameter_labels

        # Resolve cov_mode from legacy flag
        if cov_mode is None and return_cov:
            cov_mode = 'stderr' #default covariance is stderr, i.e. np.sqrt(np.diag(cov))
        if cov_mode is not None:
            return_cov = True  # enforce

        # Update metadata
        self.metadata["fit_parameters"].update({
        "use_kmeans": use_kmeans,
        "n_clusters": n_clusters if use_kmeans else None,
        "loss": loss,
        "f_scale": f_scale,
        "return_cov": return_cov,
        "cov_mode": cov_mode,
        "return_metrics": bool(return_metrics)})

        # Ensure guesses are available (lazy dask is fine)
        if guesses is None:
            guesses = self.do_kmeans_guess(n_clusters) if use_kmeans else self.do_guess()

        # Determine output size
        # Params (N) + optional covariance payload + optional metrics (2)
        cov_size = 0
        if cov_mode is None:
            cov_size = 0
        elif cov_mode == 'full':
            cov_size = self.num_params ** 2
        elif cov_mode in ('diag', 'stderr'):
            cov_size = self.num_params
        else:
            raise ValueError(f"Unknown cov_mode={cov_mode!r}. Use None, 'full', 'diag', or 'stderr'.")

        metrics_size = 4 if return_metrics else 0
        out_dim = self.num_params + cov_size + metrics_size


        def fit_worker(data_block, guess_block, x_in, ind_dims, num_params):
            data_block = np.asarray(data_block)
            guess_block = np.asarray(guess_block)

            flat_data, spat_shape = self._prepare_block(data_block, ind_dims)
            flat_guess = guess_block.reshape(-1, guess_block.shape[-1])

            out_flat = np.zeros((flat_data.shape[0], out_dim), dtype=np.float64)
            for i in range(flat_data.shape[0]):
                if flat_data[i].size == 0:
                    continue
                out_flat[i] = self._fit_logic(
                    flat_data[i],
                    x_in,
                    flat_guess[i],
                    loss=loss,
                    f_scale=f_scale,
                    return_cov=return_cov,
                    cov_mode=cov_mode,
                    return_metrics=return_metrics
                )
            return out_flat.reshape(spat_shape + (out_dim,))

        # Blockwise setup
        data_ind = tuple(range(self.ndim))
        guess_ind = tuple(self.spat_dims + [self.ndim])  # spatial dims + param dim

        self.fit_result = da.blockwise(
            fit_worker, guess_ind,
            self.dask_data, data_ind,
            guesses, guess_ind,
            self.x_axis, None,
            self.ind_dims, None,
            self.num_params, None,
            dtype=np.float64, align_arrays=True, concatenate=True
        )

        computed_result = self.fit_result.compute()
       
        return self.transform_to_sidpy(computed_result)


    def transform_to_sidpy(self, fit_dask_array):
        """
        Convert the fit results into sidpy.Dataset(s).
        Handles splitting parameters, covariance (optional), and metrics (optional).

        Output layout:
        - first num_params channels: fitted parameters
        - optional covariance payload:
            cov_mode='full'   -> next N*N channels (reshaped to N x N)
            cov_mode='diag'   -> next N channels
            cov_mode='stderr' -> next N channels
        - optional metrics payload (2):
            [r2, rmse]
        """
        fp = self.metadata.get("fit_parameters", {})
        cov_mode = fp.get("cov_mode", None)
        return_metrics = bool(fp.get("return_metrics", False))

        total_channels = fit_dask_array.shape[-1]

        # sizes
        cov_size = 0
        if cov_mode is None:
            cov_size = 0
        elif cov_mode == 'full':
            cov_size = self.num_params ** 2
        elif cov_mode in ('diag', 'stderr'):
            cov_size = self.num_params
        else:
            raise ValueError(f"Unknown cov_mode={cov_mode!r}. Use None, 'full', 'diag', or 'stderr'.")

        metrics_size = 2 if return_metrics else 0
        expected = self.num_params + cov_size + metrics_size

        if total_channels != expected:
            raise ValueError(
                f"Unexpected output channel count: got {total_channels}, expected {expected} "
                f"(num_params={self.num_params}, cov_mode={cov_mode}, return_metrics={return_metrics})."
            )

        # --- 1) Params dataset ---
        params_array = fit_dask_array[..., :self.num_params]
        sid_params = sid.Dataset.from_array(params_array, title=f"{self.dataset.title}_fit_params")

        def set_spatial_dims(dset):
            for i, orig_dim in enumerate(self.spat_dims):
                try:
                    orig_axis = self.dataset._axes[orig_dim].copy()
                except Exception:
                    orig_axis = sid.Dimension(np.arange(self.dataset.shape[orig_dim]),
                                            name=f"dim_{orig_dim}", dimension_type='spatial')
                dset.set_dimension(i, orig_axis)

        set_spatial_dims(sid_params)

        custom_labels = getattr(self, 'fit_parameter_labels', None)
        if custom_labels is not None and len(custom_labels) == self.num_params:
            p_qty = 'Label'
            self.metadata["fit_parameters"].update({"fit_parameter_labels": custom_labels})
        else:
            p_qty = 'index'

        p_vals = np.arange(self.num_params)
        sid_params.set_dimension(
            len(self.spat_dims),
            sid.Dimension(p_vals, name='fit_parameters', quantity=p_qty, dimension_type='spectral')
        )

        sid_params.metadata = dict(self.metadata).copy()
        sid_params.provenance = {'sidpy': {'generated_from': self.dataset.title}}
        sid_params.data_type = 'image_stack'

        out = [sid_params]
        cursor = self.num_params

        # --- 2) Metrics dataset ---
        if return_metrics:
            metrics_payload = fit_dask_array[..., cursor:cursor + 4]
            cursor += 4

            sid_metrics = sid.Dataset.from_array(metrics_payload, title=f"{self.dataset.title}_fit_metrics")
            set_spatial_dims(sid_metrics)
            sid_metrics.set_dimension(
                len(self.spat_dims),
                sid.Dimension(np.arange(4), name='metrics', quantity='index', dimension_type='spectral')
            )
            sid_metrics.metadata = dict(self.metadata).copy()
            sid_metrics.metadata.setdefault("fit_parameters", {}).update({"metrics": ["r2", "rmse", "aic", "bic"]})
            sid_metrics.provenance = {'sidpy': {'generated_from': self.dataset.title, 'parent_fit': sid_params.title}}
            out.append(sid_metrics)


        # --- 3) Covariance dataset (optional) ---
        if cov_size > 0:
            cov_payload = fit_dask_array[..., cursor:cursor + cov_size]
            cursor += cov_size

            if cov_mode == 'full':
                cov_shape = fit_dask_array.shape[:-1] + (self.num_params, self.num_params)
                cov_array = cov_payload.reshape(cov_shape)
                sid_cov = sid.Dataset.from_array(cov_array, title=f"{self.dataset.title}_fit_covariance")
                set_spatial_dims(sid_cov)
                sid_cov.set_dimension(
                    len(self.spat_dims),
                    sid.Dimension(p_vals, name='parameter_row', quantity=p_qty, dimension_type='spectral')
                )
                sid_cov.set_dimension(
                    len(self.spat_dims) + 1,
                    sid.Dimension(p_vals, name='parameter_col', quantity=p_qty, dimension_type='spectral')
                )
            else:
                title = f"{self.dataset.title}_fit_cov_{cov_mode}"
                sid_cov = sid.Dataset.from_array(cov_payload, title=title)
                set_spatial_dims(sid_cov)
                sid_cov.set_dimension(
                    len(self.spat_dims),
                    sid.Dimension(p_vals, name='fit_parameters', quantity=p_qty, dimension_type='spectral')
                )

            sid_cov.metadata = dict(self.metadata).copy()
            sid_cov.provenance = {'sidpy': {'generated_from': self.dataset.title, 'parent_fit': sid_params.title}}
            out.append(sid_cov)

        return out[0] if len(out) == 1 else tuple(out)


    

    
    
