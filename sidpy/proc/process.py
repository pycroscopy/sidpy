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
            cov_results = None

        elif len(self.fit_results[0]) == 2:
            # here we get back both: the parameter means and the covariance matrix!
            mean_fit_results = np.squeeze(np.array([self.fit_results[ind][0] for ind in range(len(self.fit_results))]))
            cov_results = np.squeeze(np.array([self.fit_results[ind][1] for ind in range(len(self.fit_results))]))
            # cov_results_reshaped_shape = self.pos_dim_shapes + cov_results[0].shape
            # self.cov_results = np.array(cov_results).reshape(cov_results_reshaped_shape)
        else:
            raise ValueError('Your fit function returned more than two arrays. This is not supported. Only return \
                  optimized parameters and covariance matrix')

        # Here we have either the mean fit results or both mean and cov arrays. We make 2 sidpy dataset out of them
        # Make a sidpy dataset
        mean_sid_dset = Dataset.from_array(mean_fit_results, title='Fitting_Map')
        mean_sid_dset.metadata['fold_attr'] = self._unfold_attr.copy()
        mean_sid_dset = mean_sid_dset.unfold()

        # Set the data type
        mean_sid_dset.data_type = self.dataset.data_type  # We may want to pass a new type - fit map

        # Add quantity and units
        mean_sid_dset.units = self.dataset.units
        mean_sid_dset.quantity = self.dataset.quantity

        # We set the last dimension, i.e., the dimension with the fit parameters
        fit_dim = [Dimension(np.arange(self.num_fit_parms),
                             name='fit_parms', units='a.u.',
                             quantity='fit_parameters',
                             dimension_type='spectral')]
        mean_fit_results.set_dimension(len(mean_sid_dset), fit_dim)

        # Here we deal with the covariance dataset
        if cov_results is not None:
            # Make a sidpy dataset
            cov_sid_dset = Dataset.from_array(self.cov_results, title='Fitting_Map_Covariance')
            num_fit_parms = mean_fit_results.shape[-1]
            fold_attr = self._unfold_attr.copy()
            fold_attr['dim_order_flattened'] = fold_attr['dim_order_flattened'] + [
                len(fold_attr['dim_order_flattened'])]
            fold_attr['shape_transposed'] = fold_attr['shape_transposed'] + [num_fit_parms] + [num_fit_parms]

            cov_sid_dset.metadata['fold_attr'] = fold_attr
            cov_sid_dset = cov_sid_dset.unfold()

            # Set the data type
            cov_sid_dset.data_type = self.dataset.data_type  # We may want to pass a new type - fit map

            # Add quantity and units
            cov_sid_dset.units = self.dataset.units
            cov_sid_dset.quantity = self.dataset.quantity

            cov_dims = [Dimension(np.arange(self.num_fit_parms),
                                  name='fit_cov_parms_x', units='a.u.',
                                  quantity='fit_cov_parameters',
                                  dimension_type='spectral'),
                        Dimension(np.arange(self.num_fit_parms),
                                  name='fit_cov_parms_y', units='a.u.',
                                  quantity='fit_cov_parameters',
                                  dimension_type='spectral')]

            for i, dim in enumerate(cov_dims):
                cov_sid_dset.set_dimension(i + len(cov_sid_dset.shape), dim)

        return mean_sid_dset
