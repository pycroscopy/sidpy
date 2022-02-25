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
        for i in np.arange(self.dataset.ndim):
            if i not in self.ind_dims:
                fold_order[0].extend([i])
                self.num_computations *= self.dataset.shape[i]
            else:
                fold_order.append([i])

        self.folded_dataset = self.dataset.fold(dim_order=fold_order)
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
        self.guess_results_arr = np.array(self.guess_results)

        self.guess_results_reshaped_shape = self.pos_dim_shapes + tuple([-1])
        self.guess_results_reshaped = np.array(self.guess_results_arr).reshape(self.guess_results_reshaped_shape)
        num_model_parms = self.guess_results_reshaped.shape[-1]
        self.prior = self.guess_results_reshaped.reshape(-1, num_model_parms)
        self.guess_completed = True
        return self.guess_results, self.guess_results_reshaped




