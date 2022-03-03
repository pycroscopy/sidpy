from dask.distributed import Client
import numpy as np
import dask
from ..sid import Dimension, Dataset
from ..sid.dimension import DimensionType
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import inspect


class SidFitter():
    # An extension of the Process Class for Functional Fitting
    def __init__(self, xvec, sidpy_dataset, ind_dims, fit_fn, guess_fn=None, num_fit_parms=None,
                 fit_parameter_labels=None, num_workers=2, threads=2):
        """
        Parameters
        ----------
        xvec: (numpy ndarray) Independent variable for fitting. Should be an array

        sidpy_dataset: (sidpy.Dataset) Sidpy dataset object to be fit

        ind_dims: (tuple) Tuple with integer entries of the dimensions
            over which to parallelize. These should be the independent variable for the fitting.

        fit_fn: (function) Function used for fitting.

        guess_fn: (function) (optional) This optional function should be utilized to generate priors for the full fit
        It takes the same arguments as the fitting function and should return the same type of results array.

        If the guess_fn is NOT provided, then the user MUST input the num_fit_parms.

        num_fit_parms: (int) Number of fitting parameters. This is needed IF the guess function is not provided.

        fit_parameter_labels: (list) (default None) List of parameter labels

        num_workers: (int) (default =2) Number of workers to use when setting up Dask client

        threads: (int) (default =2) Number of threads to use when setting up Dask client

        """
        if guess_fn is None:
            if num_fit_parms is None:
                raise ValueError("You did not supply a guess function, you must atleast provide number of fit "
                                 "parameters")

        self.guess_completed = False
        self.num_fit_parms = num_fit_parms
        self.xvec = xvec
        self.dataset = sidpy_dataset
        self.fit_fn = fit_fn
        self.ind_dims = ind_dims
        self._setup_calc()
        self.guess_fn = guess_fn
        self.prior = None
        self.fit_labels = fit_parameter_labels
        self.guess_results = []
        self.fit_results = []
        self.num_workers = num_workers
        self.threads = threads

        # set up dask client
        self.client = Client(threads_per_worker=self.threads, n_workers=self.num_workers)

    def _setup_calc(self):
        fold_order = [[]]  # All the independent dimensions go into the first element and will be collapsed
        self.num_computations = 1

        # Here we have to come up with a way that treats the spatial dimensions as the independent dimensions
        # In other words make the argument 'ind_dims' optional
        # if self.ind_dims is not None:

        for i in np.arange(self.dataset.ndim):
            if i in self.ind_dims:
                fold_order[0].extend([i])
                self.num_computations *= self.dataset.shape[i]
            else:
                fold_order.append([i])

        self.folded_dataset = self.dataset.fold(dim_order=fold_order)
        # Here is the tricky part, dataset.unfold is designed to get back the original dataset with minimal loss of
        # information. To do this, unfold utilizes the saved information of the original dataset. Now we are going to
        # tweak that information and use the unfold function operate after the fitting.

        self._unfold_attr = {'dim_order_flattened': fold_order[0] + [len(fold_order)],
                             'shape_transposed': [self.dataset.shape[i] for i in fold_order[0]] + [-1]}
        axes = self.dataset._axes.copy()
        axes.popitem()
        self._unfold_attr['_axes'] = axes

        self.fit_results = []

    def do_guess(self):
        """
        Get back to this
        Returns
        -------

        """
        self.guess_results = []
        for ind in range(self.num_computations):
            lazy_result = dask.delayed(self.guess_fn)(self.xvec, self.dataset_flat[ind, :])
            self.guess_results.append(lazy_result)

        self.guess_results = dask.compute(*self.guess_results)
        self.prior = np.squeeze(np.array(self.guess_results))
        self.num_fit_parms = self.prior.shape[-1]
        self.guess_completed = True

    def do_fit(self, **kwargs):
        """
        Perform the fit.
        **kwargs: extra parameters passed to the fitting function, e.g. bounds, type of lsq algorithm, etc.
        """
        if self.guess_completed is False and self.guess_fn is not None:
            # Calling the guess function
            self.do_guess()

        self.fit_results = []

        for ind in range(self.num_computations):
            if self.prior is None:
                p0 = np.random.normal(loc=0.5, scale=0.1, size=self.num_fit_parms)
            else:
                p0 = self.prior[ind, :]

            lazy_result = dask.delayed(self.fit_fn)(self.xvec, self.dataset_flat[ind, :], p0=p0, **kwargs)
            self.fit_results.append(lazy_result)

        self.fit_results = dask.compute(*self.fit_results)

        if len(self.fit_results[0]) == 1:
            # in this case we can just dump it to an array because we only got the parameters back
            mean_fit_results = np.squeeze(np.array(self.fit_results))
            self.cov_results = []

        elif len(self.fit_results[0]) == 2:
            # here we get back both: the parameter means and the covariance matrix!
            mean_fit_results = np.array([self.fit_results[ind][0] for ind in range(len(self.fit_results))])
            cov_results = np.array([self.fit_results[ind][1] for ind in range(len(self.fit_results))])
            cov_results_reshaped_shape = self.pos_dim_shapes + cov_results[0].shape
            self.cov_results = np.array(cov_results).reshape(cov_results_reshaped_shape)
        else:
            raise ValueError('Your fit function returned more than two arrays. This is not supported. Only return \
                  optimized parameters and covariance matrix')

        self.fit_opt_results_shape = self.pos_dim_shapes + tuple([-1])
        self.fit_opt_results = mean_fit_results.reshape(self.fit_opt_results_shape)

        self.num_fit_parms = self.fit_opt_results.shape[-1]

        # get it into a sidpy dataset
        dims_new_pos = [self.dataset._axes[self.pos_dims[tup_val]] for tup_val in self.pos_dims]

        # that gives a list of the dimensions we can copy into the fit dataset
        # we need the last dimension which is just the fit parameters

        fit_dims = [sid.Dimension(np.arange(self.num_fit_parms),
                                  name='fit_parms', units='a.u.',
                                  quantity='fit_parameters',
                                  dimension_type='spectral')]

        cov_dims = [sid.Dimension(np.arange(self.num_fit_parms),
                                  name='fit_cov_parms_x', units='a.u.',
                                  quantity='fit_cov_parameters',
                                  dimension_type='spectral'),
                    sid.Dimension(np.arange(self.num_fit_parms),
                                  name='fit_cov_parms_y', units='a.u.',
                                  quantity='fit_cov_parameters',
                                  dimension_type='spectral')]

        fit_dataset_dims = dims_new_pos + fit_dims
        fit_cov_dims = dims_new_pos + cov_dims

        # Specify dimensions

        # Make a sidpy dataset
        mean_sid_dset = sid.Dataset.from_array(self.fit_opt_results, name='Fitting_Map')

        # Set the data type
        mean_sid_dset.data_type = self.dataset.data_type  # We may want to pass a new type - fit map

        # Add quantity and units
        mean_sid_dset.units = self.dataset.units
        mean_sid_dset.quantity = self.dataset.quantity

        # Add dimension info
        for ind, val in enumerate(fit_dataset_dims):
            mean_sid_dset.set_dimension(ind, val)

        # append metadata
        original_parms_dict = self.dataset.metadata
        fit_func_str = str(self.fit_fn)
        guess_func_str = str(self.guess_fn)

        fit_parms_dict = {'original_metadata': original_parms_dict,
                          'fitting method': fit_func_str,
                          'guess_method': guess_func_str, 'fitting_dimensions': self.pos_dims,
                          'fit_parameter_labels': self.fit_labels}

        mean_sid_dset.metadata = fit_parms_dict

        if len(self.cov_results) > 0:
            # Make a sidpy dataset
            cov_sid_dset = sid.Dataset.from_array(self.cov_results, name='Fitting_Map_Covariance')

            # Set the data type
            cov_sid_dset.data_type = self.dataset.data_type  # We may want to pass a new type - fit map

            # Add quantity and units
            cov_sid_dset.units = self.dataset.units
            cov_sid_dset.quantity = self.dataset.quantity

            # Add dimension info
            for ind, val in enumerate(fit_cov_dims):
                cov_sid_dset.set_dimension(ind, val)

            fit_results_final = [mean_sid_dset, cov_sid_dset]
        else:
            fit_results_final = [mean_sid_dset]

        # We have a list of sidpy dataset objects
        # But what we also want is the original sidpy dataset object, amended with these results as attributes
        if not hasattr(self.dataset, 'fit_results'):
            self.dataset.fit_results = []
            self.dataset.fit_vectors = []  # list of svectors
            self.dataset.guess_fns = []
            self.dataset.fit_fns = []

        self.dataset.metadata = fit_parms_dict
        self.dataset.fit_results.append(fit_results_final)
        self.dataset.fit_vectors.append(self.xvec)
        self.dataset.guess_fns.append(self.guess_fn)
        self.dataset.fit_fns.append(self.fit_fn)

        return fit_results_final, self.dataset
