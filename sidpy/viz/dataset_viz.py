# -*- coding: utf-8 -*-
"""
Utilities for generating static image and line plots of near-publishable quality

Created on Thu May 05 13:29:12 2020

@author: Gerd Duscher
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.patches as patches
import matplotlib.animation as animation

from ..sid.dataset import Dataset

if sys.version_info.major == 3:
    unicode = str

default_cmap = plt.cm.viridis


class CurveVisualizer(object):
    def __init__(self, dset, ref_dims, figure =None,**kwargs):

        if not isinstance(dset, Dataset):
            raise TypeError('dset should be a sidpy.Dataset object')

        fig_args = dict()
        temp = kwargs.pop('figsize', None)
        if temp is not None:
            fig_args['figsize'] = temp
        print(figure)
        if figure == None:
            self.fig = plt.figure(**fig_args)
        else:
            self.fig = figure

        self.dset = dset

        # Handle the simple cases first:
        fig_args = dict()
        temp = kwargs.pop('figsize', None)
        if temp is not None:
            fig_args['figsize'] = temp

        if len(ref_dims) != 1:
            print( 'data type not handled yet')
        self.dim = self.axes[ref_dims[0]]
        if False:#is_complex_dtype(np.array(dset)):
            # Plot real and image
            fig, axes = plt.subplots(nrows=2, **fig_args)

            for axis, ufunc, comp_name in zip(axes.flat, [np.abs, np.angle], ['Magnitude', 'Phase']):
                axis.plot(self.dset.dims[ref_dims][0], ufunc(np.squeeze(curve)), **kwargs)
                if comp_name == 'Magnitude':
                    axis.set_title(self.dset.file.filename.split('/')[-1] + '\n(' + comp_name + ')', pad=15)
                    axis.set_xlabel(self.dset.get_dimension_labels()[ref_dims[0]])# + x_suffix)
                    axis.set_ylabel(self.dset.data_descriptor)
                    axis.ticklabel_format(style='sci', scilimits=(-2, 3))
                else:
                    axis.set_title(comp_name, pad=15)
                    axis.set_ylabel('Phase (rad)')
                    axis.set_xlabel(self.get_dimension_labels()[ref_dims[0]])# + x_suffix)
                    axis.ticklabel_format(style='sci', scilimits=(-2, 3))

            fig.tight_layout()
            return fig, axes

        else:
            self.axis = self.fig.add_subplot(1,1,1, **fig_args)
            self.axis.plot(self.dim.values, self.dset, **kwargs)
            self.axis.set_title(self.dset.title, pad=15)
            self.axis.set_xlabel('{}, [{}]'.format(self.dim.label, self.dim.units))# + x_suffix)
            self.axis.set_ylabel('{}, [{}]'.format(self.dset.quantity, self.dset.units))
            self.axis.ticklabel_format(style='sci', scilimits=(-2, 3))
            self.fig.canvas.draw_idle()


class ImageVisualizer(object):
    """
    Interactive display of image plot

    The stack can be scrolled through with a mouse wheel or the slider
    The ususal zoom effects of matplotlib apply.
    Works on every backend because it only depends on matplotlib.

    Important: keep a reference to this class to maintain interactive properties so usage is:

    >>view = plot_stack(dataset, {'spatial':[0,1], 'stack':[2]})

    Input:
    ------
    - dset: NSIDask _dataset
    - dim_dict: dictionary
        with key: "spatial" list of int: dimension of image
    """
    def __init__(self, dset,  figure =None,**kwargs):
        """
        plotting of data according to two axis marked as 'spatial' in the dimensions
        """
        if not isinstance(dset, Dataset):
            raise TypeError('dset should be a sidpy.Dataset object')

        fig_args = dict()
        temp = kwargs.pop('figsize', None)
        if temp is not None:
            fig_args['figsize'] = temp

        if figure is None:
            self.fig = plt.figure(**fig_args)
        else:
            self.fig = figure

        self.dset = dset

        selection = []
        image_dims = []
        for dim, axis in dset.axes.items():
            if axis.dimension_type == 'spatial':
                selection.append(slice(None))
                image_dims.append(dim)
            else:
                selection.append(slice(0, 1))
        if len(image_dims) != 2:
            raise ValueError('We need two dimensions with dimension_type spatial to plot an image')

        if False:#is_complex_dtype(self.dset):
            # Plot real and image
            fig, axes = plt.subplots(nrows=2, **fig_args)
            for axis, ufunc, comp_name in zip(axes.flat, [np.abs, np.angle], ['Magnitude', 'Phase']):
                cbar_label = self.data_descriptor
                if comp_name == 'Phase':
                    cbar_label = 'Phase (rad)'
                plot_map(axis, ufunc(np.squeeze(img)), show_xy_ticks=True, show_cbar=True,
                         cbar_label=cbar_label, x_vec=ref_dims[1].values, y_vec=ref_dims[0].values,
                         **kwargs)
                axis.set_title(self.name + '\n(' + comp_name + ')', pad=15)
                axis.set_xlabel(ref_dims[1].name + ' (' + ref_dims[1].units + ')' + suffix[1])
                axis.set_ylabel(ref_dims[0].name + ' (' + ref_dims[0].units + ')' + suffix[0])
            fig.tight_layout()
            return fig, axes

        else:

            self.axis = self.fig.add_subplot(1,1,1)
            self.axis.set_title(dset.title)
            self.img = self.axis.imshow(self.dset[tuple(selection)].T, extent=dset.get_extent(image_dims))
            self.axis.set_xlabel("{} [{}]".format(self.dset.axes[image_dims[0]].quantity, self.dset.axes[image_dims[0]].units))
            self.axis.set_ylabel("{} [{}]".format(self.dset.axes[image_dims[1]].quantity, self.dset.axes[image_dims[1]].units))

            cbar = self.fig.colorbar(self.img)
            cbar.set_label("{} [{}]".format(dset.quantity, dset.units))

            self.axis.ticklabel_format(style='sci', scilimits=(-2, 3))
            self.fig.tight_layout()
            self.img.axes.figure.canvas.draw_idle()


class ImageStackVisualizer(object):
    """
    Interactive display of image stack plot

    The stack can be scrolled through with a mouse wheel or the slider
    The ususal zoom effects of matplotlib apply.
    Works on every backend because it only depends on matplotlib.

    Important: keep a reference to this class to maintain interactive properties so usage is:

    >>view = plot_stack(dataset, {'spatial':[0,1], 'stack':[2]})

    Input:
    ------
    - dset: NSI_dataset
    - dim_dict: dictionary
        with key: "spatial" list of int: dimension of image
        with key: "time" or "stack": list of int: dimension of image stack

    """
    def __init__(self, dset, figure =None,**kwargs):
        if not isinstance(dset, Dataset):
            raise TypeError('dset should be a sidpy.Dataset object')
        fig_args = dict()
        temp = kwargs.pop('figsize', None)
        if temp is not None:
            fig_args['figsize'] = temp


        if figure == None:
            self.fig = plt.figure(**fig_args)
        else:
            self.fig = figure


        if dset.ndim <3:
            raise KeyError('dataset must have at least three dimensions')
            return
        stack_dim = -1
        image_dims=[]
        for dim, axis in dset.axes.items():
            if axis.dimension_type == 'spatial':
                image_dims.append(dim)
            else:
                stack_dim = dim
        if len(image_dims)<2:
            raise KeyError('spatial key in dimension_dictionary must be list of length 2')
            return

        ### We need one stack dimension and two image dimensions as lists in dictionary
        if stack_dim< 0:
            raise KeyError('dimension_dictionary must contain a spatial key')
            return


        if stack_dim != 0 or image_dims != [1,2]:
            ## axes not in expected order, displaying a copy of data with right dimensional oreder:
            self.cube =  np.transpose(dset, (stack_dim[0], image_dims[0],image_dims[1]))
        else:
            self.cube  = dset
        self.dset = dset

        self.axis = self.fig.add_axes([0.0, 0.2, .9, .7])
        self.ind = 0
        self.img = self.axis.imshow(self.cube[self.ind].T, extent=dset.get_extent(image_dims), **kwargs )

        interval = 100 # ms, time between animation frames

        self.number_of_slices= self.cube.shape[0]

        self.axis.set_title('image stack: '+dset.title+'\n use scroll wheel to navigate images')
        self.img.axes.figure.canvas.mpl_connect('scroll_event', self._onscroll)
        self.axis.set_xlabel("{} [{}]".format(self.dset.axes[image_dims[0]].quantity, self.dset.axes[image_dims[0]].units))
        self.axis.set_ylabel("{} [{}]".format(self.dset.axes[image_dims[1]].quantity, self.dset.axes[image_dims[1]].units))
        cbar = self.fig.colorbar(self.img)
        cbar.set_label("{} [{}]".format(dset.quantity, dset.units))

        axidx = self.fig.add_axes([0.1, 0.05, 0.55, 0.03])
        self.slider = Slider(axidx, 'image', 0, self.cube.shape[0]-1, valinit=self.ind, valfmt='%d')
        self.slider.on_changed(self._onSlider)
        playax = self.fig.add_axes([0.7, 0.05, 0.09, 0.03])
        self.play_button = Button(playax, 'Play')#, hovercolor='0.975')

        self.play = False


        self.play_button.on_clicked(self._play_slice)

        sumax = self.fig.add_axes([0.8, 0.05, 0.09, 0.03])
        self.sum_button = Button(sumax, 'Average')#, hovercolor='0.975')
        self.sum_button.on_clicked(self._sum_slice)
        self.sum = False

        self.anim = animation.FuncAnimation(self.fig, self._updatefig, interval=200, blit=False, repeat = True)
        self._update()

    def _sum_slice(self,event):
        self.img.set_data(np.average(self.cube, axis = 0).T)
        self.img.axes.figure.canvas.draw_idle()

    def _play_slice(self,event):
        self.play = not self.play
        if self.play:
            self.anim.event_source.start()
        else:
            self.anim.event_source.stop()

    def _onSlider(self, val):
        self.ind = int(self.slider.val+0.5)
        self.slider.valtext.set_text('{}'.format(self.ind))
        self._update()

    def _onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.number_of_slices
        else:
            self.ind = (self.ind - 1) % self.number_of_slices
        self.ind = int(self.ind)
        self.play = False
        self.anim.event_source.stop()
        self.slider.set_val(self.ind)

    def _update(self):
        self.img.set_data(self.cube[int(self.ind)].T)
        self.img.axes.figure.canvas.draw_idle()
        if not self.play:
            self.anim.event_source.stop()

    def _updatefig(self,*args):
        self.ind = (self.ind+1) % self.number_of_slices
        self.slider.set_val(self.ind)

        return self.img


class SpectralImageVisualizer(object):

    """
    ### Interactive spectrum imaging plot

    """

    def __init__(self, dset,  figure =None, horizontal = True, **kwargs):
        if not isinstance(dset, Dataset):
            raise TypeError('dset should be a sidpy.Dataset object')

        fig_args = dict()
        temp = kwargs.pop('figsize', None)
        if temp is not None:
            fig_args['figsize'] = temp

        if figure == None:
            self.fig = plt.figure(**fig_args)
        else:
            self.fig = figure

        if len(dset.shape) <3:
            raise KeyError('dataset must have at least three dimensions')
            return

        # We need one stack dim and two image dimes as lists in dictionary

        selection = []
        image_dims = []
        spectral_dims = []
        for dim, axis in dset.axes.items():
            if axis.dimension_type == 'spatial':
                selection.append(slice(None))
                image_dims.append(dim)
            elif axis.dimension_type == 'spectral':
                selection.append(slice(0,1))
                spectral_dims.append(dim)
            else:
                selection.append(slice(0, 1))
        if len(image_dims) != 2:
            raise ValueError('We need two dimensions with dimension_type spatial to plot an image')


        if len(image_dims) !=2 :
            raise KeyError('spatial key in dimension_dictionary must be list of length 2')
            return

        if len(spectral_dims) != 1:
            raise KeyError('spectral key in dimension_dictionary must be list of length 1')
            return

        extent = dset.get_extent([image_dims[0],image_dims[1]])

        self.horizontal = horizontal
        self.x = 0
        self.y = 0
        self.bin_x = 1
        self.bin_y = 1

        sizeX = dset.shape[image_dims[0]]
        sizeY = dset.shape[image_dims[1]]


        self.dset =dset
        self.energy_scale = dset.energy_scale.values

        self.extent = [0,sizeX,sizeY,0]
        self.rectangle = [0,sizeX,0,sizeY]
        self.scaleX = 1.0
        self.scaleY = 1.0
        self.analysis = []
        self.plot_legend = False

        self.image_dims = image_dims
        self.spec_dim = spectral_dims[0]

        if horizontal:
            self.axes = self.fig.subplots(ncols=2)
        else:
            self.axes = self.fig.subplots(nrows=2, **fig_args)

        self.fig.canvas.set_window_title(self.dset.title)
        self.image = np.average(self.dset, axis=spectral_dims[0])

        self.axes[0].imshow(self.image.T, extent = self.extent, **kwargs)
        if horizontal:
            self.axes[0].set_xlabel('{} [pixels]'.format(self.dset.axes[image_dims[0]].quantity))
        else:
            self.axes[0].set_ylabel('{} [pixels]'.format(self.dset.axes[image_dims[1]].quantity))
        self.axes[0].set_aspect('equal')

        #self.rect = patches.Rectangle((0,0),1,1,linewidth=1,edgecolor='r',facecolor='red', alpha = 0.2)
        self.rect = patches.Rectangle((0,0),self.bin_x,self.bin_y,linewidth=1,edgecolor='r',facecolor='red', alpha = 0.2)

        self.axes[0].add_patch(self.rect)
        self.intensity_scale = 1.
        self.spectrum = self.get_spectrum()

        self.axes[1].plot(self.energy_scale,self.spectrum)
        self.axes[1].set_title('spectrum {}, {}'.format(self.x, self.y))
        self.xlabel = "{}  [{}]".format(self.dset.energy_scale.quantity, self.dset.energy_scale.units)
        self.axes[1].set_xlabel(self.xlabel)# + x_suffix)
        self.ylabel = "{}  [{}]".format(self.dset.quantity, self.dset.units)
        self.axes[1].set_ylabel(self.ylabel)
        self.axes[1].ticklabel_format(style='sci', scilimits=(-2, 3))
        self.fig.tight_layout()
        self.cid = self.axes[1].figure.canvas.mpl_connect('button_press_event', self._onclick)

        self.fig.canvas.draw_idle()

    def set_bin(self,bin):

        old_bin_x = self.bin_x
        old_bin_y = self.bin_y
        if isinstance(bin, list):

            self.bin_x = int(bin[0])
            self.bin_y = int(bin[1])

        else:
            self.bin_x = int(bin)
            self.bin_y = int(bin)

        if self.bin_x > self.dset.shape[self.image_dims[0]]:
            self.bin_x = self.dset.shape[self.image_dims[0]]
        if self.bin_y > self.dset.shape[self.image_dims[1]]:
            self.bin_y = self.dset.shape[self.image_dims[1]]

        self.rect.set_width(self.rect.get_width()*self.bin_x/old_bin_x)
        self.rect.set_height((self.rect.get_height()*self.bin_y/old_bin_y))
        if self.x+self.bin_x >  self.dset.shape[self.image_dims[0]]:
            self.x = self.dset.shape[0]-self.bin_x
        if self.y+self.bin_y >  self.dset.shape[self.image_dims[1]]:
            self.y = self.dset.shape[1]-self.bin_y

        self.rect.set_xy([self.x*self.rect.get_width()/self.bin_x +  self.rectangle[0],
                            self.y*self.rect.get_height()/self.bin_y +  self.rectangle[2]])
        self._update()

    def get_spectrum(self):

        if self.x > self.dset.shape[self.image_dims[0]]-self.bin_x:
            self.x = self.dset.shape[self.image_dims[0]]-self.bin_x
        if self.y > self.dset.shape[self.image_dims[1]]-self.bin_y:
            self.y = self.dset.shape[self.image_dims[1]]-self.bin_y
        selection = []

        for dim, axis in self.dset.axes.items():
            if axis.dimension_type == 'spatial':
                if dim == self.image_dims[0]:
                    selection.append(slice(self.x, self.x + self.bin_x))
                else:
                    selection.append(slice(self.y, self.y + self.bin_y))

            elif axis.dimension_type == 'spectral':
                selection.append(None)
            else:
                selection.append(slice(0, 1))

        self.spectrum = np.squeeze(np.average(self.dset[tuple(selection)], axis=(self.image_dims)) )
        # * self.intensity_scale[self.x,self.y]
        return   np.squeeze(self.spectrum)

    def _onclick(self,event):
        self.event = event
        if event.inaxes in [self.axes[0]]:
            x = int(event.xdata)
            y = int(event.ydata)

            x= int(x - self.rectangle[0])
            y= int(y - self.rectangle[2])

            if x>=0 and y>=0:
                if x<=self.rectangle[1] and y<= self.rectangle[3]:
                    self.x = int(x/(self.rect.get_width()/self.bin_x))
                    self.y = int(y/(self.rect.get_height()/self.bin_y))

                    if self.x+self.bin_x >  self.dset.shape[0]:
                        self.x = self.dset.shape[0]-self.bin_x
                    if self.y+self.bin_y >  self.dset.shape[1]:
                        self.y = self.dset.shape[1]-self.bin_y

                    self.rect.set_xy([self.x*self.rect.get_width()/self.bin_x +  self.rectangle[0],
                                      self.y*self.rect.get_height()/self.bin_y +  self.rectangle[2]])
        # self.ax1.set_title(f'{self.x}')
        self._update()

    def _update(self, ev=None):

        xlim = self.axes[1].get_xlim()
        ylim = self.axes[1].get_ylim()
        self.axes[1].clear()
        self.get_spectrum()

        self.axes[1].plot(self.energy_scale,self.spectrum, label = 'experiment')

        self.axes[1].set_title('spectrum {}, {}'.format(self.x, self.y))


        self.axes[1].set_xlim(xlim)
        self.axes[1].set_ylim(ylim)
        self.axes[1].set_xlabel(self.xlabel)
        self.axes[1].set_ylabel(self.ylabel)

        self.fig.canvas.draw_idle()

    def set_legend(self, setLegend):
        self.plot_legend = setLegend

    def get_xy(self):
        return [self.x,self.y]
