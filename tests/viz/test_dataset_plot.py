"""
Created on Thurs Dec 10 2021

@author: Gerd Duscher
"""

import unittest
import sys
import os

import ipywidgets
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
mpl.use('Agg')
import numpy as np
sys.path.insert(0, "../../sidpy/")
import sidpy

fitter_present = True
try: 
    from sidpy.proc.fitter import SidFitter
except:
    fitter_present = False
    
def get_fit_dataset(dset_shape=(5,5,32)):
    #Define the function we want each spectrum to

    def one_lin_func(xvec, *coeff):
        a1,a2 = coeff
        return a1*xvec + a2 


    #create a dataset
    xvec = np.linspace(0,1, dset_shape[-1])
    data_mat = np.zeros(shape=(dset_shape[0]*dset_shape[1], dset_shape[2]))
    noise_level = 0.10

    for xind in range(data_mat.shape[0]):
        y_values = one_lin_func(xvec, *[np.random.uniform(0,1), np.random.normal()]) + \
        noise_level*np.random.normal(size=len(xvec))
        data_mat[xind] = y_values
        
    data_mat = data_mat.reshape(dset_shape)

    #make it a sidpy dataset
    data_set = sidpy.Dataset.from_array(data_mat, name='test_dataset')
    data_set.data_type = 'spectral_image' 
    data_set.units = 'nA'
    data_set.quantity = 'Current'

    data_set.set_dimension(0, sidpy.Dimension(np.arange(data_set.shape[0]),
                                            name='x', units='um', quantity='Length',
                                            dimension_type='spatial'))
    data_set.set_dimension(1, sidpy.Dimension(np.arange(data_set.shape[0]),
                                            'y', units='um', quantity='Length',
                                            dimension_type='spatial'))
    data_set.set_dimension(2, sidpy.Dimension(xvec,
                                            name = 'bias',quantity = 'V', units = 'V', dimension_type='spectral'))
    fitter = SidFitter(data_set, one_lin_func,num_workers=4,
                           threads=2, return_cov=False, return_fit=True, return_std=False,
                           km_guess=False,num_fit_parms = 2)
    output = fitter.do_fit()
    return data_set, output[0], output[1]
 
    

 
def get_spectrum(dtype=float):
    x = np.array(np.random.normal(3, 2.5, size=1024), dtype=dtype)

    dset = sidpy.Dataset.from_array(x)

    # dataset metadata
    dset.data_type = 'spectrum'
    dset.title = 'random'
    dset.quantity = 'intensity'
    dset.units = 'a.u.'
    scale = .5
    offset = 390
    dset.set_dimension(0, sidpy.Dimension(np.arange(dset.shape[0]) * scale + offset, 'energy'))
    dset.dim_0.dimension_type = 'spectral'
    dset.energy.units = 'eV'
    dset.energy.quantity = 'energy'
    return dset, x


def get_image(dtype=float):
    x = np.array(np.random.normal(3, 2.5, size=(512, 512)), dtype=dtype)
    dset = sidpy.Dataset.from_array(x)
    dset.data_type = 'image'
    dset.units = 'counts'
    dset.quantity = 'intensity'
    dset.title = 'random'
    dset.set_dimension(0, sidpy.Dimension(np.arange(dset.shape[0]) * .02, 'x'))
    dset.x.dimension_type = 'spatial'
    dset.x.units = 'nm'
    dset.x.quantity = 'distance'
    dset.set_dimension(1, sidpy.Dimension(np.arange(dset.shape[1]) * .02, 'y'))
    dset.y.dimension_type = 'spatial'
    dset.y.units = 'nm'
    dset.y.quantity = 'distance'

    return dset, x


def get_image_stack(dtype=float):
    x = np.array(np.random.normal(3, 2.5, size=(25, 512, 512)), dtype=dtype)

    dset = sidpy.Dataset.from_array(x)
    dset.data_type = 'image_stack'
    dset.units = 'counts'
    dset.quantity = 'intensity'

    dset.set_dimension(0, sidpy.Dimension(np.arange(dset.shape[0]), 'frame'))
    dset.frame.dimension_type = 'temporal'
    dset.set_dimension(1, sidpy.Dimension(np.arange(dset.shape[1]) * .02, 'x'))
    dset.x.dimension_type = 'spatial'
    dset.x.units = 'nm'
    dset.x.quantity = 'distance'
    dset.set_dimension(2, sidpy.Dimension(np.arange(dset.shape[2]) * .02, 'y'))
    dset.y.dimension_type = 'spatial'
    dset.y.units = 'nm'
    dset.y.quantity = 'distance'

    return dset, x


def get_spectral_image(dtype=float):

    x = np.array(np.random.normal(3, 2.5, size=(25, 512, 512)), dtype=dtype)

    dset = sidpy.Dataset.from_array(x)
    dset.data_type = 'spectral_image'
    dset.units = 'counts'
    dset.quantity = 'intensity'

    dset.set_dimension(0, sidpy.Dimension(np.arange(dset.shape[0]), 'energy'))
    dset.energy.dimension_type = 'spectral'
    dset.energy.units = 'eV'
    dset.energy.quantity = 'energy'

    dset.set_dimension(1, sidpy.Dimension(np.arange(dset.shape[1]) * .02, 'x'))
    dset.x.dimension_type = 'spatial'
    dset.x.units = 'nm'
    dset.x.quantity = 'distance'
    dset.set_dimension(2, sidpy.Dimension(np.arange(dset.shape[2]) * .02, 'y'))
    dset.y.dimension_type = 'spatial'
    dset.y.units = 'nm'
    dset.y.quantity = 'distance'

    return dset, x

def get_point_cloud(dtype=float):
    data = np.array(np.random.normal(3, 2.5, size=(20, 10)), dtype=dtype)
    data_var = np.array(np.random.normal(10, 2.5, size=(20, 10)), dtype=dtype)
    coordinates = np.array(np.random.rand(20, 2) + 10, dtype=dtype)

    dset = sidpy.Dataset.from_array(data, coordinates=coordinates)
    dset.data_type = 'point_cloud'

    dset.variance = data_var
    dset.point_cloud['spacial_units'] = 'um'
    dset.point_cloud['quantity'] = 'Distance'

    dset.set_dimension(0, sidpy.Dimension(np.arange(data.shape[0]),
                                          name='point number',
                                          quantity='Point number',
                                          dimension_type='point_cloud'))

    dset.set_dimension(1, sidpy.Dimension(np.arange(data.shape[1]),
                                          name='X',
                                          units='a.u.',
                                          quantity='X',
                                          dimension_type='spectral'))
    dset.units = 'a.u.'
    dset.quantity = 'Intensity'
    return dset, data





def get_4d_image(dtype=float):

    data = np.array(np.random.random([5, 5, 10, 10]), dtype=dtype)
    for i in range(5):
        for j in range(5):
            data[i, j] += (i+j)

    dataset = sidpy.Dataset.from_array(data)
    dataset.data_type = 'Image_4d'
    dataset.title = 'random'

    dataset.set_dimension(0, sidpy.Dimension(np.arange(dataset.shape[0]) * .02, 'u'))
    dataset.u.dimension_type = 'reciprocal'
    dataset.u.units = '1/nm'
    dataset.u.quantity = 'frequency'
    dataset.set_dimension(1, sidpy.Dimension(np.arange(dataset.shape[1]) * .02, 'v'))
    dataset.v.dimension_type = 'reciprocal'
    dataset.v.units = '1/nm'
    dataset.v.quantity = 'frequency'

    dataset.set_dimension(2, sidpy.Dimension(np.arange(dataset.shape[2]) * .02, 'x'))
    dataset.x.dimension_type = 'spatial'
    dataset.x.units = 'nm'
    dataset.x.quantity = 'distance'
    dataset.set_dimension(3, sidpy.Dimension(np.arange(dataset.shape[3]) * .02, 'y'))
    dataset.y.dimension_type = 'spatial'
    dataset.y.units = 'nm'
    dataset.y.quantity = 'distance'

    return dataset, data




class TestSpectrumPlot(unittest.TestCase):

    def test_spectrum(self):
        # dimension with metadata
        dset, x = get_spectrum()
        view = dset.plot(verbose=True)

        x_y = view.axes[0].lines[0].get_xydata()
        self.assertTrue(np.allclose(x_y[:, 1], x))
        # TODO: Fix this test
        # self.assertTrue(np.allclose(x_y[:, 0], dset.energy))
        self.assertEqual(dset.title, view.axes[0].get_title())
        self.assertEqual(f"{dset.energy.quantity} ({dset.energy.units})", view.axes[0].get_xlabel())

    def test_false_type(self):
        x = np.zeros(5)
        with self.assertRaises(TypeError):
            sidpy.viz.dataset_viz.CurveVisualizer(x)

    def test_false_dim(self):
        dset, x = get_image()
        with self.assertRaises(TypeError):
            sidpy.viz.dataset_viz.CurveVisualizer(dset)

    def test_generic(self):
        x = np.random.normal(3, 2.5, size=(4, 1024))
        dset = sidpy.Dataset.from_array(x)
        dset.data_type = 'spectrum'
        dset.plot()

    def test_complex(self):
        dset, x = get_spectrum(dtype=complex)
        view = dset.plot(verbose=True)

        x_y = view.axes[0].lines[0].get_xydata()
        self.assertEqual(len(view.axes), 2)

        self.assertTrue(np.allclose(x_y[:, 1], np.abs(x)))
        # TODO: Fix this test
        # self.assertTrue(np.allclose(x_y[:, 0], dset.energy))
        # self.assertEqual(dset.title, view.axes[0].get_title())
        self.assertEqual(f"{dset.energy.quantity} ({dset.energy.units})", view.axes[0].get_xlabel())


class TestImagePlot(unittest.TestCase):

    def test_image(self):
        dset, x = get_image()
        view = dset.plot()

        data = view.axes[0].images[0].get_array().data

        self.assertTrue(np.allclose(data.shape, x.shape))

        self.assertEqual(dset.title, view.axes[0].get_title())
        self.assertEqual(f"{dset.x.quantity} ({dset.x.units})", view.axes[0].get_xlabel())

    def test_false_type(self):
        x = np.zeros(5)
        with self.assertRaises(TypeError):
            sidpy.viz.dataset_viz.ImageVisualizer(x)

    def test_false_dim(self):
        dset, x = get_spectrum()
        with self.assertRaises(TypeError):
            sidpy.viz.dataset_viz.ImageVisualizer(dset)

    def test_generic(self):
        x = np.random.normal(3, 2.5, size=(6, 1024))
        dset = sidpy.Dataset.from_array(x)
        dset.data_type = 'image'
        dset.dim_0.dimension_type = 'spatial'
        dset.dim_1.dimension_type = 'spatial'

        dset.data_type = 'image'

        dset.plot()

    def test_complex(self):
        dset, x = get_image(dtype=complex)
        view = dset.plot(verbose=True)

        x_y = view.axes[0].images[0].get_array().data
        self.assertEqual(len(view.axes), 4)

        self.assertEqual(x_y.shape, x.shape)
        # self.assertEqual(dset.title, view.axes[0].get_title())
        self.assertEqual(f"{dset.x.quantity} ({dset.x.units})", view.axes[0].get_xlabel())

    def test_image_scale(self):
        dset, x = get_image()
        kwargs = {'scale_bar': True, 'cmap': 'hot'}  # or 'cmap': 'gray'
        view = dset.plot(verbose=True, **kwargs)

        data = view.axes[0].images[0].get_array().data

        self.assertTrue(np.allclose(data.shape, x.shape))

        self.assertEqual(dset.title, view.axes[0].get_title())
        self.assertEqual(f"{dset.x.quantity} ({dset.x.units})", view.axes[0].get_xlabel())

    def test_image_stack(self):
        dset, x = get_image_stack()
        view = sidpy.viz.dataset_viz.ImageVisualizer(dset)
        data = view.fig.axes[0].images[0].get_array().data

        self.assertTrue(np.allclose(data.shape, x.shape[1:]))

        self.assertEqual('generic_image 0', view.fig.axes[0].get_title())
        self.assertEqual(f"{dset.x.quantity} ({dset.x.units})", view.fig.axes[0].get_xlabel())



class TestImageStackPlot(unittest.TestCase):

    def test_plot(self):
        dset, x = get_image_stack()
        view = dset.plot()

        data = view.axes[0].images[0].get_array().data

        self.assertTrue(np.allclose(data.shape, x.shape[1:]))
        self.assertEqual(view.axes[0].get_title(), 'Image stack: generic\n use scroll wheel to navigate images')

        self.assertEqual(f"{dset.x.quantity} ({dset.x.units})", view.axes[0].get_xlabel())

    def test_scalebar(self):
        dset, x = get_image_stack()
        kwargs = {'scale_bar': True, 'cmap': 'hot'}  # or 'cmap': 'gray'
        view = dset.plot(verbose=True, **kwargs)
        data = view.axes[0].images[0].get_array().data
        self.assertTrue(np.allclose(data.shape, x.shape[1:]))

    def test_false_type(self):
        x = np.zeros(5)
        with self.assertRaises(TypeError):
            sidpy.viz.dataset_viz.ImageStackVisualizer(x)

    def test_false_dim(self):
        dset, x = get_spectrum()
        with self.assertRaises(TypeError):
            sidpy.viz.dataset_viz.ImageStackVisualizer(dset)

    def test_button_up(self):
        dset, x = get_image_stack()
        viz = sidpy.viz.dataset_viz.ImageStackVisualizer(dset)
        viz.slider.value = 2
        data = viz.fig.axes[0].images[0].get_array().data
        self.assertTrue(np.allclose(data.shape, x.shape[1:]))

    def test_button_average(self):
        dset, x = get_image_stack()
        viz = sidpy.viz.dataset_viz.ImageStackVisualizer(dset)
        viz.button.value = True
        data = viz.fig.axes[0].images[0].get_array().data
        self.assertTrue(np.allclose(data.shape, x.shape[1:]))

    def test_button_average2(self):
        dset, x = get_image_stack()
        viz = sidpy.viz.dataset_viz.ImageStackVisualizer(dset)
        viz.button.value = True
        viz.button.value = False
        data = viz.fig.axes[0].images[0].get_array().data
        self.assertTrue(np.allclose(data.shape, x.shape[1:]))


class TestSpectralImagePlot(unittest.TestCase):

    def test_plot(self):
        dset, x = get_spectral_image()
        view = dset.plot()
        self.assertEqual(len(view.axes), 2)

    def test_bin(self):
        dset, x = get_spectral_image()
        view = dset.plot()

        dset.view.set_bin([20, 20])

        self.assertEqual(len(view.axes), 2)

        dset.view.set_bin(10)
        self.assertEqual(len(view.axes), 2)

    def test_false_type(self):
        x = np.zeros(5)
        with self.assertRaises(TypeError):
            sidpy.viz.dataset_viz.SpectralImageVisualizer(x)

    def test_false_dim(self):
        dset, x = get_spectrum()
        with self.assertRaises(TypeError):
            sidpy.viz.dataset_viz.SpectralImageVisualizer(dset)
        dset, x = get_image()
        with self.assertRaises(TypeError):
            sidpy.viz.dataset_viz.SpectralImageVisualizer(dset)

        dset, x = get_4d_image()
        with self.assertRaises(TypeError):
            sidpy.viz.dataset_viz.SpectralImageVisualizer(dset)

class TestPointCloudPlot(unittest.TestCase):
    def test_plot_basic(self):
        dset, x = get_point_cloud()
        view = dset.plot()
        self.assertEqual(len(view.axes), 2)
        self.assertEqual(view.axes[0].get_xlabel(), 'Distance [px]')
        self.assertEqual(view.axes[0].get_ylabel(), 'Distance [px]')

        self.assertEqual(view.axes[1].get_xlabel(), 'X (a.u.)')
        self.assertEqual(view.axes[1].get_ylabel(), 'Intensity (a.u.)')

        x_y = view.axes[1].lines[0].get_xydata()
        self.assertEqual(x_y.shape, (10, 2))

    def test_false_type(self):
        x = np.zeros(5)
        with self.assertRaises(TypeError):
            sidpy.viz.dataset_viz.PointCloudVisualizer(x)

    def test_false_dim(self):
        dset, x = get_spectrum()
        with self.assertRaises(TypeError):
            sidpy.viz.dataset_viz.PointCloudVisualizer(dset)
        dset, x = get_image()
        with self.assertRaises(TypeError):
            sidpy.viz.dataset_viz.PointCloudVisualizer(dset)
        dset, x = get_4d_image()
        with self.assertRaises(TypeError):
            sidpy.viz.dataset_viz.PointCloudVisualizer(dset)
        dset, x = get_spectral_image()
        with self.assertRaises(TypeError):
            sidpy.viz.dataset_viz.PointCloudVisualizer(dset)

    def test_units_button(self):
        dset, x = get_point_cloud()
        view = dset.plot()
        self.assertIsInstance(dset.view.button, ipywidgets.Dropdown)
        self.assertEqual(dset.view.button.value, 1)

        dset.view.button.value = 2
        self.assertEqual(view.axes[0].get_xlabel(), 'Distance [um]')
        self.assertEqual(view.axes[0].get_ylabel(), 'Distance [um]')

    def test_point_selection(self):
        dset, x = get_point_cloud()
        view = dset.plot()

        event = mpl.backend_bases.MouseEvent(
            name='button_press_event',
            canvas=dset.view.fig.canvas,
            x=0,  # x-coordinate of the click (adjust as needed)
            y=0,  # y-coordinate of the click (adjust as needed)
            button=1,  # button number (1 for left button)
        )
        xpos, ypos = 25, 25
        event.inaxes = dset.view.axes[0]
        event.xdata = xpos
        event.ydata = ypos
        dset.view._onclick(event)

        selected_point = dset.view.tree.query(np.array([xpos, ypos]))[1]
        spectrum_title = dset.view.axes[1].get_title()
        self.assertEqual(spectrum_title, 'point {}'.format(selected_point))
        actual = dset.view.axes[1].lines[0].get_ydata()
        expected = dset[selected_point].compute()
        self.assertTrue(np.allclose(actual, expected, equal_nan=True, rtol=1e-05, atol=1e-08))


class Test4DImageStackPlot(unittest.TestCase):

    def test_plot(self):
        dataset, data = get_4d_image()
        view = dataset.plot()
        self.assertEqual(len(view.axes), 2)

    def test_bin(self):
        dset, x = get_4d_image()
        view = dset.plot()

        dset.view.set_bin([20, 20])

        self.assertEqual(len(view.axes), 2)

        dset.view.set_bin(10)
        self.assertEqual(len(view.axes), 2)

    def test_scan_directions(self):
        dataset, data = get_4d_image()
        view = dataset.plot(scan_x=3,scan_y=2, image_4d_x=1, image_4d_y=0)
        self.assertEqual(len(view.axes), 2)

    def test_false_type(self):
        x = np.zeros(5)
        with self.assertRaises(TypeError):
            sidpy.viz.dataset_viz.FourDimImageVisualizer(x)

    def test_false_dim(self):
        dset, x = get_image()
        with self.assertRaises(TypeError):
            sidpy.viz.dataset_viz.FourDimImageVisualizer(dset)

        dset, x = get_spectral_image()
        with self.assertRaises(TypeError):
            sidpy.viz.dataset_viz.FourDimImageVisualizer(dset)

    def test_plot_complex(self):
        dataset, data = get_4d_image(dtype=complex)
        view = dataset.plot()
        self.assertEqual(len(view.axes), 3)



class TestSpectralImageFitVisualizer(unittest.TestCase):

    def test_plot_with_fit_parms(self): 
        if fitter_present:           
             original_dataset, fit_parameters, fitted_dataset = get_fit_dataset()
             view = sidpy.viz.dataset_viz.SpectralImageFitVisualizer(original_dataset, fit_parameters)
             self.assertEqual(len(view.axes), 2)

    def test_plot_with_fitted_dataset(self):
        if fitter_present: 
            original_dataset, fit_parameters, fitted_dataset = get_fit_dataset()
            view = sidpy.viz.dataset_viz.SpectralImageFitVisualizer(original_dataset, fitted_dataset)
            self.assertEqual(len(view.axes), 2)
        
    def test_plot_with_custom_xvec(self):
        if fitter_present: 
            original_dataset, fit_parameters, fitted_dataset = get_fit_dataset()
            xvec = np.linspace(-1,2,32)
            view = sidpy.viz.dataset_viz.SpectralImageFitVisualizer(original_dataset, fit_parameters, xvec = xvec)
            self.assertEqual(len(view.axes), 2)
    
if __name__ == '__main__':
    unittest.main()
