import numpy as np
from astropy.visualization import LogStretch, LinearStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from .map_plotter import MapPlotter, MapsPlotter

__metaclass__ = type

class MeshPlotter(MapPlotter):

    def plot_mesh(self, data, x=None, y=None, contours=None, levels=None, 
            colors='g'):#, extent=None):

        # Normalisation
        if self.stretch=='log':
            norm = ImageNormalize(vmin=self.vmin, vmax=self.vmax, 
                    stretch=LogStretch(a=self.a))
        else:
            norm = ImageNormalize(vmin=self.vmin, vmax=self.vmax, 
                    stretch=LinearStretch())

        # Plot data
        if x is None and y is None:
            y, x = np.indices(data.shape) - 0.5

        self.im = self.ax.pcolormesh(x, y, data, norm=norm, zorder=1,
                edgecolors='face', rasterized=True)
        #self.im = self.ax.imshow(data, norm=norm, zorder=1, extent=extent,
        #        interpolation='nearest')

        # Plot contours
        if contours is not None and levels is not None:
            if contours_wcs is not None:
                self.ax.contour(contours, levels=levels, colors=colors, 
                        zorder=2)
            else:
                self.ax.contour(contours, levels=levels, colors=colors,
                        zorder=2)

    def plot_field(self, x, y, vx, vy, scale=10000., scale_units='dots',
            **kwargs):
        self.ax.quiver(x, y, vx, vy, scale_units=scale_units, scale=scale,
                **kwargs)
        #self.ax.set_aspect('equal', 'datalim')

    def config_plot(self, xlim=None, ylim=None, xlabel=None, ylabel=None,
            set_xticklabels=True, set_yticklabels=True, title=None):
        if xlim:
            self.ax.set_xlim(*xlim)
        if ylim:
            self.ax.set_ylim(*ylim)
        if xlabel:
            self.ax.set_xlabel(xlabel)
        if ylabel:
            self.ax.set_ylabel(ylabel)
        if not set_xticklabels:
            self.ax.xaxis.set_ticklabels(['']*len(self.ax.xaxis.get_ticklabels()))
        if not set_yticklabels:
            self.ax.yaxis.set_ticklabels(['']*len(self.ax.yaxis.get_ticklabels()))
        if title:
            self.set_title(title)

class NMeshPlotter(MapsPlotter):

    def get_mesh(self, n, vmin=0, vmax=None, a=1000, stretch=None,
            projection=None, include_cbar=True):
        vmin = vmin or self.vmin
        vmax = vmax or self.vmax
        a = a or self.a
        stretch = stretch or self.stretch
        axis, cbax = self.get_axis(n, projection=projection,
                include_cbar=include_cbar)
        return MeshPlotter(axis, cbax, vmin=vmin, vmax=vmax, a=a, stretch=stretch)
