
#Define the fit functions for loop fitting and SHO fitting
import os
import tempfile
import unittest
import logging

from pathlib import Path
from urllib.request import urlretrieve
import numpy as np
import matplotlib.pyplot as plt
import sys
import dask.array as da
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.special import erf, erfinv


#For testing sidpy fitter refactored
def loop_fit_function(vdc, *coef_vec):
    """
    9 parameter fit function

    Parameters
    -----------
    vdc : 1D numpy array or list
        DC voltages
    coef_vec : 1D numpy array or list
        9 parameter coefficient vector

    Returns
    ---------
    loop_eval : 1D numpy array
        Loop values
    """

    vdc = np.array(vdc).squeeze()
    a = coef_vec[:5]
    b = coef_vec[5:]
    d = 1000

    v1 = np.asarray(vdc[:int(len(vdc) / 2)])
    v2 = np.asarray(vdc[int(len(vdc) / 2):])

    g1 = (b[1] - b[0]) / 2 * (erf((v1 - a[2]) * d) + 1) + b[0]
    g2 = (b[3] - b[2]) / 2 * (erf((v2 - a[3]) * d) + 1) + b[2]

    y1 = (g1 * erf((v1 - a[2]) / g1) + b[0]) / (b[0] + b[1])
    y2 = (g2 * erf((v2 - a[3]) / g2) + b[2]) / (b[2] + b[3])

    f1 = a[0] + a[1] * y1 + a[4] * v1
    f2 = a[0] + a[1] * y2 + a[4] * v2

    loop_eval = np.hstack((f1, f2))
   
    loop_eval = loop_eval.squeeze()
    return loop_eval

def calculate_loop_centroid(vdc, loop_vals):
    """
    Calculates the centroid of a single given loop. Uses polyogonal centroid, 
    see wiki article for details.
    
    Parameters
    -----------
    vdc : 1D list or numpy array
        DC voltage steps
    loop_vals : 1D list or numpy array
        unfolded loop
    
    Returns
    -----------
    cent : tuple
        (x,y) coordinates of the centroid
    area : float
        geometric area
    """

    vdc = np.squeeze(np.array(vdc))
    num_steps = vdc.size

    x_vals = np.zeros(num_steps - 1)
    y_vals = np.zeros(num_steps - 1)
    area_vals = np.zeros(num_steps - 1)

    for index in range(num_steps - 1):
        x_i = vdc[index]
        x_i1 = vdc[index + 1]
        y_i = loop_vals[index]
        y_i1 = loop_vals[index + 1]

        x_vals[index] = (x_i + x_i1) * (x_i * y_i1 - x_i1 * y_i)
        y_vals[index] = (y_i + y_i1) * (x_i * y_i1 - x_i1 * y_i)
        area_vals[index] = (x_i * y_i1 - x_i1 * y_i)

    area = 0.50 * np.sum(area_vals)
    cent_x = (1.0 / (6.0 * area)) * np.sum(x_vals)
    cent_y = (1.0 / (6.0 * area)) * np.sum(y_vals)

    return (cent_x, cent_y), area


def generate_guess(vdc, pr_vec, show_plots=False):
    """
    Given a single unfolded loop and centroid return the intial guess for the fitting.
    We generate most of the guesses by looking at the loop centroid and looking
    at the nearest intersection points with the loop, which is a polygon.

    Parameters
    -----------
    vdc : 1D numpy array
        DC offsets
    pr_vec : 1D numpy array
        Piezoresponse or unfolded loop
    show_plots : Boolean (Optional. Default = False)
        Whether or not the plot the convex hull, centroid, intersection points

    Returns
    -----------------
    init_guess_coef_vec : 1D Numpy array
        Fit guess coefficient vector
    """

    points = np.transpose(np.array([np.squeeze(vdc), pr_vec]))  # [points,axis]

    geom_centroid, geom_area = calculate_loop_centroid(points[:, 0], points[:, 1])

    hull = ConvexHull(points)

    """
    Now we need to find the intersection points on the N,S,E,W
    the simplex of the complex hull is essentially a set of line equations.
    We need to find the two lines (top and bottom) or (left and right) that
    interect with the vertical / horizontal lines passing through the geometric centroid
    """

    def find_intersection(A, B, C, D):
        """
        Finds the coordinates where two line segments intersect

        Parameters
        ------------
        A, B, C, D : Tuple or 1D list or 1D numpy array
            (x,y) coordinates of the points that define the two line segments AB and CD

        Returns
        ----------
        obj : None or tuple
            None if not intersecting. (x,y) coordinates of intersection
        """

        def ccw(A, B, C):
            """Credit - StackOverflow"""
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def line(p1, p2):
            """Credit - StackOverflow"""
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0] * p2[1] - p2[0] * p1[1])
            return A, B, -C

        def intersection(L1, L2):
            """
            Finds the intersection of two lines (NOT line segments).
            Credit - StackOverflow
            """
            D = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]
            if D != 0:
                x = Dx / D
                y = Dy / D
                return x, y
            else:
                return None

        if ((ccw(A, C, D) is not ccw(B, C, D)) and (ccw(A, B, C) is not ccw(A, B, D))) is False:
            return None
        else:
            return intersection(line(A, B), line(C, D))

    # start and end coordinates of each line segment defining the convex hull
    outline_1 = np.zeros((hull.simplices.shape[0], 2), dtype=float)
    outline_2 = np.zeros((hull.simplices.shape[0], 2), dtype=float)
    for index, pair in enumerate(hull.simplices):
        outline_1[index, :] = points[pair[0]]
        outline_2[index, :] = points[pair[1]]

    """Find the coordinates of the points where the vertical line through the
    centroid intersects with the convex hull"""
    y_intersections = []
    for pair in range(outline_1.shape[0]):
        x_pt = find_intersection(outline_1[pair], outline_2[pair],
                                 [geom_centroid[0], hull.min_bound[1]],
                                 [geom_centroid[0], hull.max_bound[1]])
        if x_pt is not None:
            y_intersections.append(x_pt)

    '''
    Find the coordinates of the points where the horizontal line through the
    centroid intersects with the convex hull
    '''
    x_intersections = []
    for pair in range(outline_1.shape[0]):
        x_pt = find_intersection(outline_1[pair], outline_2[pair],
                                 [hull.min_bound[0], geom_centroid[1]],
                                 [hull.max_bound[0], geom_centroid[1]])
        if x_pt is not None:
            x_intersections.append(x_pt)

    '''
    Default values if not intersections can be found.
    '''
    if len(y_intersections) < 2:
        min_y_intercept = min(pr_vec)
        max_y_intercept = max(pr_vec)
    else:
        min_y_intercept = min(y_intersections[0][1], y_intersections[1][1])
        max_y_intercept = max(y_intersections[0][1], y_intersections[1][1])

    if len(x_intersections) < 2:
        min_x_intercept = min(vdc) / 2.0
        max_x_intercept = max(vdc) / 2.0
    else:
        min_x_intercept = min(x_intersections[0][0], x_intersections[1][0])
        max_x_intercept = max(x_intersections[0][0], x_intersections[1][0])

    # Only the first four parameters use the information from the intercepts
    # a3, a4 are swapped in Stephen's figure. That was causing the branches to swap during fitting
    # the a3, a4 are fixed now below:
    init_guess_coef_vec = np.zeros(shape=9)
    init_guess_coef_vec[0] = min_y_intercept
    init_guess_coef_vec[1] = max_y_intercept - min_y_intercept
    init_guess_coef_vec[2] = min_x_intercept
    init_guess_coef_vec[3] = max_x_intercept
    init_guess_coef_vec[4] = 0
    init_guess_coef_vec[5] = 2  # 0.5
    init_guess_coef_vec[6] = 2  # 0.2
    init_guess_coef_vec[7] = 2  # 1.0
    init_guess_coef_vec[8] = 2  # 0.2

    if show_plots:
        fig, ax = plt.subplots()
        ax.plot(points[:, 0], points[:, 1], 'o')
        ax.plot(geom_centroid[0], geom_centroid[1], 'r*')
        ax.plot([geom_centroid[0], geom_centroid[0]], [hull.max_bound[1], hull.min_bound[1]], 'g')
        ax.plot([hull.min_bound[0], hull.max_bound[0]], [geom_centroid[1], geom_centroid[1]], 'g')
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], 'k')
        ax.plot(x_intersections[0][0], x_intersections[0][1], 'r*')
        ax.plot(x_intersections[1][0], x_intersections[1][1], 'r*')
        ax.plot(y_intersections[0][0], y_intersections[0][1], 'r*')
        ax.plot(y_intersections[1][0], y_intersections[1][1], 'r*')
        ax.plot(vdc, loop_fit_function(vdc, init_guess_coef_vec))

    return init_guess_coef_vec

#----SHO Functions-----
def SHO_fit_flattened(wvec,*p):
    Amp, w_0, Q, phi=p[0],p[1],p[2],p[3]
    func = Amp * np.exp(1.j * phi) * w_0 ** 2 / (wvec ** 2 - 1j * wvec * w_0 / Q - w_0 ** 2)
    return np.hstack([np.real(func),np.imag(func)])

def sho_guess_fn(freq_vec,ydata):
    
    ydata = np.array(ydata)
    amp_guess = np.abs(ydata)[np.argmax(np.abs(ydata))]
    Q_guess = 50
    max_min_ratio = np.max(abs(ydata)) / np.min(abs(ydata))
    phi_guess = np.angle(ydata)[np.argmax(np.abs(ydata))]
    w_guess = freq_vec[np.argmax(np.abs(ydata))]

    #Let's just run some Q values to find the closest one
    Q_values = [5,10,20,50,100,200,500]
    err_vals = []
    for q_val in Q_values:
        p_test = [amp_guess/q_val, w_guess, q_val, phi_guess]
        func_out = SHO_fit_flattened(freq_vec,*p_test)
        complex_output = func_out[:len(func_out)//2] + 1j*func_out[(len(func_out)//2):]
        amp_output = np.abs(complex_output)
        err = np.mean((amp_output - np.abs(ydata))**2)
        err_vals.append(err)
    Q_guess = Q_values[np.argmin(err_vals)]
    p0 = [amp_guess/Q_guess, w_guess, Q_guess, phi_guess]
    return p0


#TODO: we only have SHO (one complex) and BEPS (non complex) datasets. 
# # Need to Test on many different cases here!

log = logging.getLogger(__name__)

class TestSidpyFitterRefactor(unittest.TestCase):
    
    def test_beps_fit(self):
        with tempfile.TemporaryDirectory() as d:
            tmp_path = Path(d)

            from sidpy.proc.fitter_refactor import SidpyFitterRefactor
            import SciFiReaders as sr
            import numpy as np

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
            

            log.info("Testing Fitting with K-Means in test_beps_fit")
            # Option 2: K-Means Prior Fitting (Better for noisy data)
            result_beps = fitter.do_fit(use_kmeans=True, n_clusters=12)
            
            #Check to see that the results metadata is beign written correctly
            model_source = result_beps.metadata['source_code']['model_function']

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

            from sidpy.proc.fitter_refactor import SidpyFitterRefactor
            import SciFiReaders as sr
            from urllib.request import urlretrieve

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



            
