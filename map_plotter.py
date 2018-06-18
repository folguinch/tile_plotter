from configparser import ConfigParser

import numpy as np
import astropy.units as u
from astropy.visualization import LogStretch, LinearStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization.wcsaxes import SphericalCircle
import matplotlib
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.patches import Rectangle, Ellipse

from .base_plotter import BasePlotter, SinglePlotter
from .functions import get_ticks

__metaclass__ = type

class MapPlotter(SinglePlotter):

    def __init__(self, ax, cbax=None, vmin=0, vmax=None, a=None,
            stretch='linear'):
        self.im = None
        self.cbax = cbax
        self.a = a
        self.vmin = vmin
        self.vmax = vmax
        self.stretch = stretch
        super(MapPlotter, self).__init__(ax)

    def plot_map(self, data, wcs=None, r=None, position=None, contours=None, 
            contours_wcs=None, levels=None, colors='g', extent=None, 
            extent_cont=None, linewidths=None, mask=False):

        # Check wcs and re-centre the image
        if wcs is not None and r is not None and position is not None:
            self.recenter(r, position[0], position[1], wcs)

        # Normalisation
        if self.stretch=='log':
            norm = ImageNormalize(vmin=self.vmin, vmax=self.vmax, 
                    stretch=LogStretch(a=self.a))
        else:
            norm = ImageNormalize(vmin=self.vmin, vmax=self.vmax, 
                    stretch=LinearStretch())

        # Plot data
        if mask:
            cmap = matplotlib.cm.get_cmap()
            cmap.set_bad('w',1.0)
            maskdata = np.ma.array(data, mask=np.isnan(data))
            self.im = self.ax.imshow(maskdata, norm=norm, zorder=1, extent=extent)
        else:
            self.im = self.ax.imshow(data, norm=norm, zorder=1, extent=extent)

        # Plot contours
        if contours is not None and levels is not None:
            self.plot_contours(contours, levels, wcs=contours_wcs,
                    extent=extent_cont, linewidths=linewidths, colors=colors,
                    zorder=2)

    def plot_contours(self, data, levels, wcs=None, extent=None, linewidths=None,
            colors='g', zorder=0):
        if wcs is not None:
            self.ax.contour(data, 
                    transform=self.ax.get_transform(wcs),
                    levels=levels, colors=colors, zorder=zorder,
                    linewidths=linewidths)
        else:
            self.ax.contour(data, levels=levels, colors=colors,
                    zorder=zorder, extent=extent, linewidths=linewidths)

    def recenter(self, r, ra, dec, wcs):
        x, y = wcs.all_world2pix([[ra,dec]], 0)[0]
        cdelt = np.mean(np.abs(wcs.wcs.cdelt))
        radius = r/cdelt
        self.ax.set_xlim(x-radius, x+radius)
        self.ax.set_ylim(y-radius, y+radius)

    def config_map(self, xformat="hh:mm:ss.s", yformat="dd:mm:ss", xlabel=True,
            ylabel=True, xticks=True, yticks=True, xpad=1., ypad=1., 
            tickscolor='w', xcoord='ra', ycoord='dec'):

        ra, dec = self.ax.coords[xcoord], self.ax.coords[ycoord]

        # Axes labels
        ra.set_axislabel('RA (J2000)' if xlabel else '', 
                size=self.ax.xaxis.get_label().get_fontsize(),
                family=self.ax.xaxis.get_label().get_family(),
                fontname=self.ax.xaxis.get_label().get_fontname(), minpad=xpad)
        dec.set_axislabel('Dec (J2000)' if ylabel else '',
                size=self.ax.xaxis.get_label().get_fontsize(),
                family=self.ax.xaxis.get_label().get_family(),
                fontname=self.ax.xaxis.get_label().get_fontname(), minpad=ypad)

        # Ticks labels
        ra.set_major_formatter(xformat)
        ra.set_ticklabel_visible(xticks)
        dec.set_major_formatter(yformat)
        dec.set_ticklabel_visible(yticks)
        ra.set_ticks(color=tickscolor, exclude_overlapping=True)
        dec.set_ticks(color=tickscolor)

        # Ticks
        ra.set_ticklabel(size=self.ax.xaxis.get_majorticklabels()[0].get_fontsize(),
                family=self.ax.xaxis.get_majorticklabels()[0].get_family(),
                fontname=self.ax.xaxis.get_majorticklabels()[0].get_fontname())
        dec.set_ticklabel(size=self.ax.yaxis.get_majorticklabels()[0].get_fontsize(),
                family=self.ax.yaxis.get_majorticklabels()[0].get_family(),
                fontname=self.ax.yaxis.get_majorticklabels()[0].get_fontname())
        ra.display_minor_ticks(True)
        dec.display_minor_ticks(True)

    def plot_cbar(self, fig, label=None, ticks=None, ticklabels=None, 
            orientation='vertical', labelpad=10):
        # Ticks
        if ticks is None:
            ticks = get_ticks(self.vmin, self.vmax, self.a, stretch=self.stretch)

        # Create bar
        cbar = fig.colorbar(self.im, ax=self.ax, cax=self.cbax,
                    orientation=orientation, drawedges=False, ticks=ticks)
            # Tick Font properties
            #print cbar.ax.get_yticklabels()[-1].get_fontsize()
            #for label in cbax.get_xticklabels()+cbax.get_yticklabels():
            #    label.set_fontsize(ax.xaxis.get_majorticklabels()[0].get_fontsize())
            #    label.set_family(ax.xaxis.get_majorticklabels()[0].get_family())
            #    label.set_fontname(ax.xaxis.get_majorticklabels()[0].get_fontname())

        # Bar position
        if orientation=='vertical':
            cbar.ax.yaxis.set_ticks_position('right')
            if ticklabels is not None:
                cbar.ax.yaxis.set_ticklabels(ticklabels)
        elif orientation=='horizontal':
            cbar.ax.xaxis.set_ticks_position('top')
            if ticklabels is not None:
                cbar.ax.xaxis.set_ticklabels(ticklabels)

        # Label
        if label is not None:
            if orientation=='vertical':
                cbar.ax.yaxis.set_label_position('right')
                cbar.set_label(label,
                        fontsize=self.ax.yaxis.get_label().get_fontsize(),
                        family=self.ax.yaxis.get_label().get_family(),
                        fontname=self.ax.yaxis.get_label().get_fontname(),
                        weight=self.ax.yaxis.get_label().get_weight(),
                        labelpad=labelpad, verticalalignment='top')
            elif orientation=='horizontal':
                cbar.ax.xaxis.set_label_position('top')
                cbar.set_label(label,
                        fontsize=self.ax.xaxis.get_label().get_fontsize(),
                        family=self.ax.xaxis.get_label().get_family(),
                        fontname=self.ax.xaxis.get_label().get_fontname(),
                        weight=self.ax.xaxis.get_label().get_weight())
        for tlabel in cbar.ax.xaxis.get_ticklabels(which='both')+cbar.ax.yaxis.get_ticklabels(which='both'):
            tlabel.set_fontsize(self.ax.xaxis.get_majorticklabels()[0].get_fontsize())
            tlabel.set_family(self.ax.xaxis.get_majorticklabels()[0].get_family())
            tlabel.set_fontname(self.ax.xaxis.get_majorticklabels()[0].get_fontname())

    def plot_beam(self, header, bmin=None, bmaj=None, bpa=0., dx=1,
            dy=1, pad=2, color='k'):
        # Beam properties
        pixsize = np.sqrt(np.abs(header['CDELT2']*header['CDELT1']))
        bmaj = header.get('BMAJ', bmaj)/pixsize
        bmin = header.get('BMIN', bmin)/pixsize
        bpa = header.get('BPA', bpa)

        # Define position
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        xmin += dx
        ymin += dy
        size = bmaj+pad

        # Plot box and beam
        #rect = Rectangle((xmin,ymin), size, size, fill=True, fc='w', zorder=3)
        beam = Ellipse((xmin+size/2., ymin+size/2.), bmin, bmaj, angle=bpa,
                zorder=4, fc=color)
        #ax.add_patch(rect)
        self.ax.add_patch(beam)

    def annotate(self, *args, **kwargs):
        self.ax.annotate(*args, **kwargs)

    def set_xlim(self, xmin=None, xmax=None):
        self.ax.set_xlim(xmin, xmax)

    def set_ylim(self, ymin=None, ymax=None):
        self.ax.set_ylim(ymin, ymax)

    def set_title(self, title):
        self.ax.set_title(title)

    def scatter(self, x, y, **kwargs):
        self.ax.scatter(x, y, transform=self.ax.get_transform('world'), **kwargs)

    def circle(self, x, y, r, color='g', facecolor='none', zorder=0):
        cir = SphericalCircle((x, y), r, edgecolor=color, facecolor=facecolor,
                transform=self.ax.get_transform('fk5'), zorder=zorder)
        self.ax.add_patch(cir)

    def plot(self, *args, **kwargs):
        kwargs['transform'] = self.ax.get_transform('world')
        self.ax.plot(*args, **kwargs)

    def plot_scale(self, size, r, distance, x=0.1, y=0.1, dy=0.01, color='g', zorder=10,
            unit=u.au, loc=3):
        length = size.to(u.arcsec) / (2*r.to(u.arcsec))
        label = distance.to(u.pc) * size.to(u.arcsec)
        label = label.value * u.au
        label = "%s" % label.to(unit)
        self.annotate('', xy=(x,y), xytext=(x+length.value, y),
                xycoords='axes fraction', arrowprops=dict(arrowstyle="|-|", 
                    facecolor=color),
                color=color)
        xmid = x + length.value/2.
        self.annotate(label, xy=(xmid,y+dy), xytext=(xmid, y+dy),
                xycoords='axes fraction', color=color)
        #bar = AnchoredSizeBar(self.ax.transData, size.to(u.deg), label, loc,
        #        color=color)
        #self.ax.add_artist(bar)


class MapsPlotter(BasePlotter):

    def __init__(self, vmin=0, vmax=None, a=1000, stretch='linear', config=None, 
            **kwargs):
        super(MapsPlotter, self).__init__(**kwargs)
        
        self.stretch = stretch
        if config is not None:
            self.vmin = vmin or config.getfloat('vmin', None)
            self.vmax = vmax or config.getfloat('vmax', None)
            self.a = a or config.getfloat('a', None)
            self.stretch = stretch or config.get('stretch', 'linear')
        else:
            self.vmin = vmin
            self.vmax = vmax
            self.a = a

    def __iter__(self):
        for i, ax in enumerate(self.axes):
            axis, cbax = self.get_axis(i)
            yield MapPlotter(axis, cbax, vmin=self.vmin, vmax=self.vmax,
                    a=self.a, stretch=self.stretch)

    def get_mapper(self, n, vmin=0, vmax=None, a=1000, stretch=None,
            projection=None, include_cbar=True):
        vmin = vmin or self.vmin
        vmax = vmax or self.vmax
        a = a or self.a
        stretch = stretch or self.stretch
        axis, cbax = self.get_axis(n, projection=projection,
                include_cbar=include_cbar)

        return MapPlotter(axis, cbax, vmin=vmin, vmax=vmax, a=a, stretch=stretch)

    def init_axis(self, n, projection='rectilinear', include_cbar=False):
        super(MapsPlotter, self).init_axis(n, projection, include_cbar)

