"""
:class:`~sidpy.proc.fitter.SidFitter` class that fits the specified dimension of a sidpy.dataset using the
user-specified fit function. An extension of scipy.optimise.curve_fit that works on sidpy.dataset

Created on Mar 9, 2022
@author: Rama Vasudevan, Mani Valleti
"""

from dask.distributed import Client
import numpy as np
import dask
import inspect
from ..sid import Dimension, Dataset
from ..sid.dimension import DimensionType

try:
    from scipy.optimize import curve_fit
except ModuleNotFoundError:
    curve_fit = None


class SidFitter:
    # An extension of the Process Class for Functional Fitting
    def __init__(self, sidpy_dataset, fit_fn, xvec=None, ind_dims=None, guess_fn=None, num_fit_parms=None,
                 return_cov=False, return_fit=False, fit_parameter_labels=None, num_workers=2, threads=2):
        """
        Parameters
        ----------
        sidpy_dataset: (sidpy.Dataset) Sidpy dataset object to be fit

        fit_fn: (function) Function used for fitting.
        Should take xvec as the first argument and parameters as the rest of the arguments.
        Should return the function value at each of the points in the xvec

        xvec: (numpy ndarray or list of numpy ndarrays) (Optional)
        Independent variable for fitting. Should be an array
        If NOT provided, the dimension arrays are assumed to be xvecs

        ind_dims: (tuple) (Optional) Tuple with integer entries of the dimensions
            over which to parallelize. These should be the independent variable for the fitting.
            If NOT provided, it is assumed that all the non-spectral dimensions are independent dimensions.

        guess_fn: (function) (optional) This optional function should be utilized to generate priors for the full fit
        It takes the same arguments as the fitting function and should return the same type of results array.
        If the guess_fn is NOT provided, then the user MUST input the num_fit_parms.

        num_fit_parms: (int) Number of fitting parameters. This is needed IF the guess function is not provided to set
        the priors for the parameters for the curve_fit function.

        return_cov: (bool) (default False) Returns the estimated covariance of fitting parameters. Confer
        scipy.optimize.curve_fit for further details

        return_fit: (bool) (default False) Returns the fitted sidpy dataset using the optimal parameters
         when set to true

        fit_parameter_labels: (list) (default None) List of parameter labels

        num_workers: (int) (default =2) Number of workers to use when setting up Dask client

        threads: (int) (default =2) Number of threads to use when setting up Dask client

        Returns:
        -------
        sidpy.dataset: if return_cov and return_fit are both set to False
        List: containing sidpy.dataset objects, if either of return_cov or return fit is set to True

        If multiple datasets are expected, the order of the returned datasets is

        [sidpy.dataset with mean parameter values,
        sidpy.dataset with estimated covariances of the fitting parameters,
        sidpy.dataset that is fit with the parameters obtained after fitting]

        """

        if guess_fn is None:
            if num_fit_parms is None:
                raise ValueError("You did not supply a guess function, you must at least provide number of fit "
                                 "parameters to set the priors for scipy.optimize.curve_fit")
        self.dataset = sidpy_dataset  # Sidpy dataset
        self.fit_fn = fit_fn  # function that takes xvec, *parameters and returns yvec at each value of xvec
        self.num_fit_parms = num_fit_parms  # int: number of fitting parameters

        if ind_dims is not None:
            self.ind_dims = ind_dims  # Tuple: containing indices of independent dimensions
        else:
            # All the dimensions that are not spectral will be considered as independent dimensions
            ind_dims = []
            for i, dim in self.dataset._axes.items():
                if dim.dimension_type != DimensionType.SPECTRAL:
                    ind_dims.extend([i])
            self.ind_dims = tuple(ind_dims)

        # Make sure there is at least one spectral dimension
        if len(self.ind_dims) == len(self.dataset.shape):
            raise NotImplementedError('No Spectral (dependent) dimensions found to fit')

        # Let's get the dependent dims here
        dep_dims = []  # Tuple: contains all the dependent dimensions. ind_dims+dep_dims = all_dims
        for d in np.arange(len(self.dataset.shape)):
            if d not in self.ind_dims:
                dep_dims.extend([d])
        self.dep_dims = tuple(dep_dims)

        # xvec is not provided
        if xvec is None:
            # 1D fit
            if len(self.dep_dims) == 1:
                dep_vec = np.array(self.dataset._axes[self.dep_dims[0]])
            # Multidimensional fit
            else:
                dep_vec = []
                for d in self.dep_dims:
                    dep_vec.append(np.array(self.dataset._axes[d]))

        # xvec is provided
        if xvec is not None:
            # 1D fit
            if len(self.dep_dims) == 1:
                if isinstance(xvec, np.ndarray):
                    dep_vec = xvec
                elif isinstance(xvec, list):
                    dep_vec = np.array(xvec)
                else:
                    raise TypeError('Please provide a np.ndarray or a list of independent vector values')
            # Multidimensional fit
            else:
                if isinstance(xvec, list) and len(xvec) == len(self.dep_dims):
                    dep_vec = xvec
                elif isinstance(xvec, list) and len(xvec) != len(self.dep_dims):
                    raise ValueError('The number of independent dimensions provided in the xvec do not match '
                                     'with the number of dependent dimensions of the dataset')
                else:
                    raise TypeError('Please provide a list of value-arrays corresponding to each dependent dimension')

        # Dealing with the meshgrid part of multidimensional fitting
        if len(self.dep_dims) > 1:
            self.dep_vec = [ar.ravel() for ar in np.meshgrid(*dep_dims)]
        else:
            self.dep_vec = dep_vec

        self._setup_calc()
        self.guess_fn = guess_fn
        self.prior = None  # shape = [num_computations, num_fitting_parms]
        self.fit_labels = fit_parameter_labels
        self.num_workers = num_workers
        self.threads = threads
        self.guess_completed = False
        self.return_fit = return_fit
        self.return_cov = return_cov

        self.mean_fit_results = []
        if self.return_cov:
            self.cov_fit_results = None
        # set up dask client
        self.client = Client(threads_per_worker=self.threads, n_workers=self.num_workers)

    def _setup_calc(self):
        self.fold_order = [[]]  # All the independent dimensions go into the first element and will be collapsed
        self.num_computations = 1

        # Here we have to come up with a way that treats the spatial dimensions as the independent dimensions
        # In other words make the argument 'ind_dims' optional
        # if self.ind_dims is not None:

        for i in np.arange(self.dataset.ndim):
            if i in self.ind_dims:
                self.fold_order[0].extend([i])
                self.num_computations *= self.dataset.shape[i]
            else:
                self.fold_order.append([i])

        self.folded_dataset = self.dataset.fold(dim_order=self.fold_order)

        # Here is the tricky part, dataset.unfold is designed to get back the original dataset with minimal loss of
        # information. To do this, unfold utilizes the saved information while folding the original dataset.
        # Here, we are going to tweak that information and use the unfold method on the dataset with fitted parameters.

        self._unfold_attr = {'dim_order_flattened': self.fold_order[0] + [len(self.fold_order[0])],
                             'shape_transposed': [self.dataset.shape[i] for i in self.fold_order[0]] + [-1]}
        axes = self.dataset._axes.copy()
        for i in self.dep_dims:
            del axes[i]
        self._unfold_attr['_axes'] = axes

    def do_guess(self):
        """
        If a guess_fn is provided: Applies the guess_fn to get priors for the fitting parameters.
        self.prior is set as the output of guess function at each of the ind_dims

        Returns:
        None
        -------

        """
        guess_results = []
        for ind in range(self.num_computations):
            lazy_result = dask.delayed(self.guess_fn)(self.dep_vec, self.folded_dataset[ind, :])
            guess_results.append(lazy_result)

        guess_results = dask.compute(*guess_results)
        self.prior = np.squeeze(np.array(guess_results))
        self.num_fit_parms = self.prior.shape[-1]
        self.guess_completed = True

    def do_fit(self, **kwargs):
        """
        Perform the fit.
        **kwargs: extra parameters passed to scipy.optimize.curve_fit, e.g. bounds, type of lsq algorithm, etc.
        """

        if not self.guess_completed and self.guess_fn is not None:
            # Calling the guess function
            print('We are now calling the guess function')
            guess_function_str = inspect.getsource(self.guess_fn)
            self.do_guess()

        fit_results = []
        for ind in range(self.num_computations):
            if self.prior is None:
                p0 = np.random.normal(loc=0.5, scale=0.1, size=self.num_fit_parms)
                guess_function_str = 'Not Provided'
            else:
                p0 = self.prior[ind, :]

            lazy_result = dask.delayed(SidFitter.default_curve_fit)(self.fit_fn, self.dep_vec,
                                                                    self.folded_dataset[ind, :],
                                                                    return_cov=self.return_cov, p0=p0, **kwargs)
            fit_results.append(lazy_result)

        fit_results_comp = dask.compute(*fit_results)

        if not self.return_cov:
            # in this case we can just dump it to an array because we only got the parameters back
            self.mean_fit_results = np.squeeze(np.array(fit_results_comp))

        else:
            # here we get back both: the parameter means and the covariance matrix!
            self.mean_fit_results = np.squeeze(
                np.array([fit_results_comp[ind][0] for ind in range(len(fit_results_comp))]))
            self.cov_fit_results = np.squeeze(
                np.array([fit_results_comp[ind][1] for ind in range(len(fit_results_comp))]))

        # Here we have either the mean fit results or both mean and cov arrays. We make 2 sidpy dataset out of them
        # Make a sidpy dataset

        mean_sid_dset = Dataset.from_array(self.mean_fit_results, title='Fitting_Map')
        mean_sid_dset.metadata['fold_attr'] = self._unfold_attr.copy()
        mean_sid_dset = mean_sid_dset.unfold()

        # Set the data type
        mean_sid_dset.data_type = 'image_stack'  # We may want to pass a new type - fit map

        # Add quantity and units
        mean_sid_dset.units = self.dataset.units
        mean_sid_dset.quantity = self.dataset.quantity

        # We set the last dimension, i.e., the dimension with the fit parameters
        fit_dim = Dimension(np.arange(self.num_fit_parms),
                            name='fit_parms', units='a.u.',
                            quantity='fit_parameters',
                            dimension_type='temporal')
        mean_sid_dset.set_dimension(len(mean_sid_dset.shape) - 1, fit_dim)

        fit_parms_dict = {'fit_parameters_labels': self.fit_labels,
                          'fitting_function': inspect.getsource(self.fit_fn),
                          'guess_function': guess_function_str,
                          'ind_dims': self.ind_dims,
                          }
        mean_sid_dset.metadata = self.dataset.metadata.copy()
        mean_sid_dset.metadata['fit_parms_dict'] = fit_parms_dict.copy()
        mean_sid_dset.original_metadata = self.dataset.original_metadata.copy()

        # Here we deal with the covariance dataset
        if self.return_cov:
            # Make a sidpy dataset
            cov_sid_dset = Dataset.from_array(self.cov_fit_results, title='Fitting_Map_Covariance')
            fold_attr = self._unfold_attr.copy()
            fold_attr['dim_order_flattened'] = fold_attr['dim_order_flattened'] + [
                len(fold_attr['dim_order_flattened'])]
            fold_attr['shape_transposed'] = fold_attr['shape_transposed'][:-1] + [self.num_fit_parms] + \
                                            [self.num_fit_parms]

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
                cov_sid_dset.set_dimension(i - 2 + len(cov_sid_dset.shape), dim)

            cov_sid_dset.metadata = self.dataset.metadata.copy()
            cov_sid_dset.metadata['fit_parms_dict'] = fit_parms_dict.copy()
            cov_sid_dset.original_metadata = self.dataset.original_metadata.copy()

        if self.return_fit:
            fit_dset = self.get_fitted_dataset()
            fit_dset.metadata['fit_parms_dict'] = fit_parms_dict.copy()

        if self.return_cov and self.return_fit:
            return [mean_sid_dset, cov_sid_dset, fit_dset]
        elif self.return_cov and not self.return_fit:
            return [mean_sid_dset, cov_sid_dset]
        elif not self.return_cov and self.return_fit:
            return [mean_sid_dset, fit_dset]
        else:
            return mean_sid_dset

    def get_fitted_dataset(self):
        """This method returns the fitted dataset using the parameters generated by the fit function"""
        fitted_dset = self.dataset.like_data(np.zeros_like(self.dataset.compute()),
                                             title_prefix='fitted_')
        fitted_dset_fold = fitted_dset.fold(dim_order=self.fold_order)

        # Here we make a roundtrip to numpy as earlier versions of dask did not support the assignments
        # of the form dask_array[2] = 1

        np_folded_arr = fitted_dset_fold.compute()
        for i in range(np_folded_arr.shape[0]):
            np_folded_arr[i] = self.fit_fn(self.dep_vec, *self.mean_fit_results[i])

        fitted_sid_dset_folded = fitted_dset_fold.like_data(np_folded_arr, title=fitted_dset_fold.title)
        fitted_sid_dset = fitted_sid_dset_folded.unfold()
        fitted_sid_dset.original_metadata = self.dataset.original_metadata.copy()

        return fitted_sid_dset

    # Might be a good idea to add a default fit function like np.polyfit or scipy.curve_fit

    @staticmethod
    def default_curve_fit(fit_fn, xvec, yvec, return_cov=True, **kwargs):
        print(kwargs)
        xvec = np.array(xvec)
        yvec = np.array(yvec)
        yvec = yvec.ravel()
        xvec = xvec.ravel()
        if curve_fit is None:
            raise ModuleNotFoundError("scipy is not installed")
        else:
            popt, pcov = curve_fit(fit_fn, xvec, yvec, **kwargs)

        if return_cov:
            return popt, pcov
        else:
            return popt