"""
:class:`~sidpy.proc.fitter.SidFitter` class that fits the specified dimension of a sidpy.dataset using the
user-specified fit function. An extension of scipy.optimise.curve_fit that works on sidpy.dataset

Created on Mar 9, 2022
@author: Rama Vasudevan, Mani Valleti
"""

from xml.dom import NotFoundErr
from dask.distributed import Client
import numpy as np
import dill
import base64
import dask
import inspect
from ..sid import Dimension, Dataset
from ..sid.dimension import DimensionType
from ..viz.dataset_viz import SpectralImageFitVisualizer
from ..sid.dataset import DataType

try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None

try:
    from sklearn.cluster import KMeans
except ModuleNotFoundError:
    KMeans = None


class SidFitter:
    # An extension of the Process Class for Functional Fitting
    def __init__(self, sidpy_dataset, fit_fn, xvec=None, ind_dims=None, guess_fn=None, num_fit_parms=None,
                 km_guess=False, n_clus=None, return_cov=False, return_std=False, return_fit=False,
                 fit_parameter_labels=None, num_workers=2, threads=2):
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
        It takes (xvec,yvec) as inputs and should return the fit parameters.
        If the guess_fn is NOT provided, then the user MUST input the num_fit_parms.

        num_fit_parms: (int) Number of fitting parameters. This is needed IF the guess function is not provided to set
        the priors for the parameters for the curve_fit function.

        km_guess: (bool) (default False) When set to True: Divides the spectra into clusters using
        sklearn.optimize.kMeans, applies the fitting function on the cluster centers,
        uses the results as priors to each spectrum of the cluster.

        n_clus: (int) (default None) Used only when km_guess is set to True. Determines the number of clusters to be
        formed for sklearn.optimize.kmeans. If not provided then n_clus = self.num_computations/100

        return_std: (bool) (default False) Returns the dataset with estimated standard deviation of the parameter
        values. Square roots of the diagonal of the covariance matrix.

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
        self._complex_data = False  # if data is complex. Will be checked during guess/fit as needed.

        if ind_dims is not None:
            self.ind_dims = tuple(ind_dims)  # Tuple: containing indices of independent dimensions
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
            self.dep_vec = [ar.ravel() for ar in np.meshgrid(*dep_vec, indexing='ij')]
        else:
            self.dep_vec = dep_vec

        self.km_guess = km_guess
        if self.km_guess:
            self.km_priors = None
            self.km_labels = None
            self.n_clus = n_clus

        self._setup_calc()
        self.guess_fn = guess_fn
        self.prior = None  # shape = [num_computations, num_fitting_parms]
        self.fit_labels = fit_parameter_labels
        self.num_workers = num_workers
        self.threads = threads
        self.guess_completed = False
        self.return_std = return_std
        self.return_cov = return_cov
        self.return_fit = return_fit
        self.fitted_dset = None


        self.mean_fit_results = []
        if self.return_cov:
            self.cov_fit_results = None
        if self.return_std:
            self.std_fit_results = None

        if 'complex' in self.dataset.dtype.name:
            self._complex_data = True
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
        self.folded_dataset_numpy = np.array(self.folded_dataset)
        self.dep_vec = np.array(self.dep_vec)

        # Here is the tricky part, dataset.unfold is designed to get back the original dataset with minimal loss of
        # information. To do this, unfold utilizes the saved information while folding the original dataset.
        # Here, we are going to tweak that information and use the unfold method on the dataset with fitted parameters.

        self._unfold_attr = {
            'dim_order_flattened': list(np.arange(len(self.fold_order[0]))) + [len(self.fold_order[0])],
            'shape_transposed': [self.dataset.shape[i] for i in self.fold_order[0]] + [-1]}
        axes, j = {}, 0
        for i, dim in self.dataset._axes.items():
            if not i in self.dep_dims:
                axes[j] = dim
                j += 1
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
            ydata = self.folded_dataset_numpy
            lazy_result = dask.delayed(self.guess_fn)(self.dep_vec, ydata[ind, :])
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
        

        if self.guess_fn is not None:
            guess_function_str = inspect.getsource(self.guess_fn)
        else:
            guess_function_str = 'Not Provided'

        fit_results = []
        if not self.km_guess:
            if not self.guess_completed and self.guess_fn is not None:
                self.do_guess()

            for ind in range(self.num_computations):
                if self.prior is None:
                    p0 = np.random.normal(loc=0.5, scale=0.1, size=self.num_fit_parms)
                else:
                    p0 = self.prior[ind, :]
                ydata = self.folded_dataset_numpy[ind, :]
                if self._complex_data:
                    ydata = np.array(np.hstack([np.real(ydata), np.imag(ydata)]))

                lazy_result = dask.delayed(SidFitter.default_curve_fit)(self.fit_fn, self.dep_vec,
                                                                        ydata, self.num_fit_parms,
                                                                        return_cov=(self.return_cov or self.return_std),
                                                                        p0=p0, **kwargs)
                fit_results.append(lazy_result)

            fit_results_comp = dask.compute(*fit_results)
            self.client.close()

        else:
            self.get_km_priors(**kwargs)
            for ind in range(self.num_computations):
                ydata = self.folded_dataset_numpy[ind, :]
                if self._complex_data:
                    #ydata = ydata.flatten_complex()
                    ydata = np.array(np.hstack([np.real(ydata), np.imag(ydata)]))

                lazy_result = dask.delayed(SidFitter.default_curve_fit)(self.fit_fn, self.dep_vec,
                                                                        ydata, self.num_fit_parms,
                                                                        return_cov=(self.return_cov or self.return_std),
                                                                        p0=self.km_priors[self.km_labels[ind]],
                                                                        **kwargs)
                fit_results.append(lazy_result)

            fit_results_comp = dask.compute(*fit_results)
            self.client.close()

        if self.return_cov or self.return_std:
            # here we get back both: the parameter means and the covariance matrix!
            self.mean_fit_results = np.squeeze(
                np.array([fit_results_comp[ind][0] for ind in range(len(fit_results_comp))]))
            self.cov_fit_results = np.squeeze(
                np.array([fit_results_comp[ind][1] for ind in range(len(fit_results_comp))]))

        else:
            # in this case we can just dump it to an array because we only got the parameters back
            self.mean_fit_results = np.squeeze(np.array(fit_results_comp))

        # Here we have either the mean fit results or both mean and cov arrays. We make 2 sidpy dataset out of them
        # Make a sidpy dataset

        mean_sid_dset = Dataset.from_array(self.mean_fit_results, title='Fitting_Map')
        mean_sid_dset.metadata['fold_attr'] = self._unfold_attr.copy()
        mean_sid_dset = mean_sid_dset.unfold()

        # Set the data type
        mean_sid_dset.data_type = 'image_stack'  # We may want to pass a new type - fit map

        # We set the last dimension, i.e., the dimension with the fit parameters
        fit_dim = Dimension(np.arange(self.num_fit_parms),
                            name='fit_parms', units='a.u.',
                            quantity='fit_parameters',
                            dimension_type='temporal')
        mean_sid_dset.set_dimension(len(mean_sid_dset.shape) - 1, fit_dim)

        fit_parms_dict = {'fit_parameters_labels': self.fit_labels,
                          'fitting_function': inspect.getsource(self.fit_fn),
                          'guess_function': guess_function_str,
                          'ind_dims': self.ind_dims
                          }
        mean_sid_dset.metadata = self.dataset.metadata.copy()
        mean_sid_dset.metadata['fit_parms_dict'] = fit_parms_dict.copy()
        mean_sid_dset.original_metadata = self.dataset.original_metadata.copy()
        #TODO: Here we want to save the fitting function as well
        #We should do this in a dictionary called 'fitting_functions'
        #Here is an example
        '''
        import dill

        # Create your dictionary
        fit_fn_dict = {'func1': self.fit_fn,'func2': another_func, ...}

        # Serialize the functions in the dictionary using dill and encode as base64
        encoded_dict = {}
        for key, value in my_dictionary.items():
            serialized_value = dill.dumps(value)
            encoded_value = base64.b64encode(serialized_value).decode('utf-8')
            encoded_dict[key] = encoded_value

        fit_data.metadata['fitting_functions'] = encoded_dict #save to the metadata
        When these files are read with the NSID reader, we should be able to use these functions out of the block.

        '''
        fit_fn_dict = {'func1': self.fit_fn}

        # Serialize the functions in the dictionary using dill and encode as base64
        encoded_dict = {}
        for key, value in fit_fn_dict.items():
            serialized_value = dill.dumps(value)
            encoded_value = base64.b64encode(serialized_value).decode('utf-8')
            encoded_dict[key] = encoded_value

        mean_sid_dset.metadata['fitting_functions'] = encoded_dict.copy() #save to the metadata

        cov_sid_dset, std_fit_dset, fit_dset = None, None, None

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
            cov_sid_dset.data_type = 'IMAGE_4D'  # We may want to pass a new type - fit map

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

        # Here is the std_dev dataset
        if self.return_std:
            self.std_fit_results = np.diagonal(self.cov_fit_results, axis1=-2, axis2=-1)
            std_fit_dset = Dataset.from_array(self.std_fit_results, title='Fitting_Map_std_dev')
            std_fit_dset.metadata['fold_attr'] = self._unfold_attr.copy()
            std_fit_dset = std_fit_dset.unfold()

            # Set the data type
            std_fit_dset.data_type = 'image_stack'  # We may want to pass a new type - fit map

            # We set the last dimension, i.e., the dimension with the fit parameters
            fit_dim = Dimension(np.arange(self.num_fit_parms),
                                name='std_dev', units='a.u.',
                                quantity='std_dev_fit_parms',
                                dimension_type='temporal')
            std_fit_dset.set_dimension(len(std_fit_dset.shape) - 1, fit_dim)

            std_fit_dset.metadata = self.dataset.metadata.copy()
            std_fit_dset.metadata['fit_parms_dict'] = fit_parms_dict.copy()
            std_fit_dset.original_metadata = self.dataset.original_metadata.copy()

        # Fitted dset
        if self.return_fit:
            fit_dset = self.get_fitted_dataset()
            fit_dset.metadata['fit_parms_dict'] = fit_parms_dict.copy()

        results = [mean_sid_dset, cov_sid_dset, std_fit_dset, fit_dset]
        inds = [True, self.return_cov, self.return_std, self.return_fit]
        results = [results[i] for i in range(len(inds)) if inds[i]]

        if len(results) == 0:
            return results[0]
        else:
            return results

    def get_fitted_dataset(self):
        """This method returns the fitted dataset using the parameters generated by the fit function"""
        fitted_dset = self.dataset.like_data(np.zeros_like(self.dataset.compute()),
                                             title_prefix='fitted_')

        fitted_dset_fold = fitted_dset.fold(dim_order=self.fold_order)
        output_shape = np.prod(fitted_dset_fold.shape[1:])
        user_folding = False
        ydata_fit = self.fit_fn(self.dep_vec, *self.mean_fit_results[0])

        # print(r"ydata shape is {} and squeezed is {}".format(ydata_fit.shape, ydata_fit.squeeze().shape))
        if ydata_fit.squeeze().shape[0] != output_shape:
            print('Shapes of output of fitting function is {} and original data is {} \
                  Reshaping output dataset. You are responsible for reshaping'.format(ydata_fit.shape[0],
                                                                                      output_shape,
                                                                                      ))

            fitted_dset_fold = self.dataset.like_data(np.zeros((fitted_dset_fold.shape[0], ydata_fit.shape[0])),
                                                      title_prefix='fitted_')
            user_folding = True
        # Here we make a roundtrip to numpy as earlier versions of dask did not support the assignments
        # of the form dask_array[2] = 1

        np_folded_arr = fitted_dset_fold.compute()
        for i in range(np_folded_arr.shape[0]):
            # ydata_fit = self.fit_fn(self.dep_vec, *self.mean_fit_results[i])
            # print('dep vec is {} and mean fit results are {}'.format(self.dep_vec,self.mean_fit_results[i]))
            fit_output = self.fit_fn(self.dep_vec, *self.mean_fit_results[i])
            # print('ydata output from fitting fn is {}'.format(fit_output))
            if fit_output.shape != np_folded_arr[i].shape:
                try:
                    np_folded_arr[i] = fit_output.reshape(np_folded_arr[i].shape)
                except:
                    print("Cannot reshape function output to retrieve fitted dataset")
            else:
                np_folded_arr[i] = fit_output

        if not user_folding:
            fitted_sid_dset_folded = fitted_dset_fold.like_data(np_folded_arr, title=fitted_dset_fold.title)
            fitted_sid_dset = fitted_sid_dset_folded.unfold()
            fitted_sid_dset.original_metadata = self.dataset.original_metadata.copy()
        else:
            fitted_sid_dset = fitted_dset_fold.like_data(np_folded_arr, title=fitted_dset_fold.title)
            fitted_sid_dset.original_metadata = self.dataset.original_metadata.copy()
        self.fitted_dset = fitted_sid_dset
        return fitted_sid_dset

    def get_km_priors(self, **kwargs):
        kwargs['maxfev'] = 100  # give a large number of tries for fitting the kmeans cluster centers

        shape = self.folded_dataset.shape  # We get the shape of the folded dataset
        # Our prior_dset will have the same shape except for the last dimension whose size will be equal to number of
        # fitting parameters
        dim_order = [[0], [i + 1 for i in range(len(shape) - 1)]]
        # We are using the fold function in case we have a multidimensional fit.
        # In that case we need all the spectral dimensions collapsed into a single dimension for kMeans
        # In case of a 1D fit the next line essentially does nothing.
        km_dset = self.folded_dataset.fold(dim_order)

        if self._complex_data:
            print('Warning: complex dataset detected. For Kmeans priors, we will treat real part only')
            km_dset = km_dset.real

        if KMeans is None:
            raise ModuleNotFoundError("sklearn is not installed")
        else:
            if self.n_clus is None:
                self.n_clus = int(self.num_computations / 100)
            km = KMeans(n_clusters=self.n_clus, random_state=0).fit(km_dset.compute())

        self.km_labels, self.km_centers = km.labels_, km.cluster_centers_
    
        if self._complex_data:
            km_dset = np.array(self.folded_dataset.fold(dim_order))
            self.km_centers = []
            # in the case of complex data, the centers have to be recomputed based on the labels
            for ind_l in range(self.n_clus):
                cent = km_dset[self.km_labels == ind_l, :]
                centroid = cent.real.mean(axis=0) + 1j*cent.imag.mean(axis=0)
                self.km_centers.append(centroid)
            self.km_centers = np.array(self.km_centers)
        print('---Finished KMeans, onto fiting each KM Center---')
        km_priors = []
        for i, cen in enumerate(self.km_centers):
            print('Fitting center {}'.format(i))
            num_start = 100 #number of times to restart the fit. For now this is fixed.

            if self.guess_fn is not None:
                p0 = self.guess_fn(self.dep_vec, cen)
            else:
                p0 = np.random.normal(loc=0.5, scale=0.1, size=self.num_fit_parms)
            if self._complex_data:
                cen = np.hstack([np.real(cen), np.imag(cen)])
            
            residuals = []
            for _ in range(num_start):
                
                popt = SidFitter.default_curve_fit(self.fit_fn, self.dep_vec, cen, self.num_fit_parms,
                                        return_cov=False,  p0 = p0,  **kwargs)
                temp_fit = self.fit_fn(self.dep_vec, *popt)
                #temp_fit = temp_fit[:len(temp_fit)//2] + 1j* temp_fit[len(temp_fit)//2 :]
                #temp_fit = np.hstack([np.real(cen), np.imag(cen)])
                #print(cen, temp_fit, cen.shape, temp_fit.shape)
                resid = cen - temp_fit
                resid_ss = np.sum(np.abs(resid@resid))
                residuals.append((popt, resid_ss))
                
            residuals = np.array(residuals, dtype = object)
            self.residuals = residuals
            min_idx = np.argmin(residuals[:,1])
            best_popt = residuals[min_idx,0]
            km_priors.append(best_popt)

        self.km_priors = np.array(km_priors)
        self.num_fit_parms = self.km_priors.shape[-1]

    def visualize_fit_results(self, figure=None, horizontal=True):
        '''
        Calls the interactive visualizer for comparing raw and fit datasets.

        Inputs:
            - figure: (Optional, default None) - handle to existing figure
            - horiziontal: (Optional, default True) - whether spectrum should be plotted horizontally

        '''
        dset_type = self.dataset.data_type
        supported_types = ['SPECTRAL_IMAGE']
        if self.fitted_dset == None:
            raise NotFoundErr("No fitted dataset found. Re-run with return_fit=True to use this feature")
        if dset_type == DataType.SPECTRAL_IMAGE:
            visualizer = SpectralImageFitVisualizer(self.dataset, self.fitted_dset,
                                                    figure=figure, horizontal=horizontal)
        else:
            raise NotImplementedError(
                "Data type is {} but currently we only support types {}".format(dset_type, supported_types))

        return visualizer

    @staticmethod
    def default_curve_fit(fit_fn, xvec, yvec, num_fit_parms, return_cov=True, **kwargs):

        yvec = np.array(yvec).ravel()
        if curve_fit is None:
            raise ModuleNotFoundError("scipy is not installed")
        else:
            try:
                popt, pcov = curve_fit(fit_fn, xvec, yvec, **kwargs)
            except:
                popt = np.zeros(num_fit_parms)
                pcov = np.zeros((num_fit_parms, num_fit_parms))
        if return_cov:
            return popt, pcov
        else:
            return popt
