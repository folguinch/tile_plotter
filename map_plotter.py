import os
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
from .utils import auto_vmin, auto_vmax, auto_levels

__metaclass__ = type

class MapPlotter(SinglePlotter):

    def __init__(self, ax, cbax=None, vmin=None, vmax=None, a=1000.,
            stretch='linear'):
        super(MapPlotter, self).__init__(ax, cbaxis=cbax)
        self.im = None
        self.label = None
        self.a = a #or config.get('a', fallback=100)
        self.vmin = vmin #or config.get('vmin', fallback=None)
        self.vmax = vmax #or config.get('vmax', fallback=None)
        self.stretch = stretch #or config.get('stretch', fallback='linear')

    @property
    def normalization(self):
        if self.stretch=='log':
            return ImageNormalize(vmin=self.vmin, vmax=self.vmax, 
                    stretch=LogStretch(a=self.a))
        else:
            return ImageNormalize(vmin=self.vmin, vmax=self.vmax, 
                    stretch=LinearStretch())

    def plot_map(self, data, wcs=None, label=None, r=None, position=None, 
            extent=None, self_contours=False, levels=None, colors='w', 
            linewidths=None, nlevels=10, mask=False, **kwargs):

        # Define default values for vmin and vmax
        if self.vmin is None:
            self.vmin = auto_vmin(data)
        if self.vmax is None:
            self.vmax = auto_vmax(data)

        # Check wcs and re-centre the image
        if wcs is not None and r is not None and position is not None:
            self.recenter(r, position, wcs)

        # Normalisation
        norm = self.normalization

        # Plot data
        self.label = label
        if mask:
            cmap = matplotlib.cm.get_cmap()
            cmap.set_bad('w',1.0)
            maskdata = np.ma.array(data, mask=np.isnan(data))
            self.im = self.ax.imshow(maskdata, norm=norm, zorder=1,
                    extent=extent, **kwargs)
        else:
            self.im = self.ax.imshow(data, norm=norm, zorder=1, extent=extent,
                    **kwargs)

        # Plot contours
        if self_contours:
            if levels is None:
                levels = auto_levels(data, n=nlevels, stretch=self.stretch, 
                        vmin=self.vmin, vmax=self.vmax)
            self.plot_contours(data, levels, wcs=wcs, extent=extent, 
                    colors=colors, zorder=2, linewidths=linewidths)
            return levels
        else:
            return None

    def plot_contours(self, data, levels, wcs=None, extent=None, colors='g', 
            zorder=0, **kwargs):
        if 'cmap' not in kwargs:
            kwargs['colors'] = colors
        elif 'norm' not in kwargs:
            kwargs['norm'] = self.normalization
        else:
            pass

        if wcs is not None:
            return super(MapPlotter, self).contour(data, 
                    transform=self.ax.get_transform(wcs), levels=levels, 
                    zorder=zorder, **kwargs)
        else:
            return super(MapPlotter, self).contour(data, levels=levels, 
                    zorder=zorder, extent=extent, **kwargs)

    def recenter(self, r, position, wcs):
        if hasattr(position, 'ra'):
            ra = position.ra.degree
            dec = position.dec.degree
        else:
            ra, dec = position
        x, y = wcs.all_world2pix([[ra,dec]], 0)[0]
        cdelt = np.mean(np.abs(wcs.wcs.cdelt))*u.deg
        if hasattr(r, 'unit'):
            radius = r.to(u.deg) / cdelt
        else:
            radius = r*u.deg / cdelt
        self.ax.set_xlim(x-radius.value, x+radius.value)
        self.ax.set_ylim(y-radius.value, y+radius.value)

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
            orientation='vertical', labelpad=10, lines=None):
        label = label or self.label
        return super(MapPlotter, self).plot_cbar(fig, self.im, label=label,
                vmin=self.vmin, vmax=self.vmax, a=self.a, stretch=self.stretch,
                ticks=ticks, ticklabels=ticklabels, orientation=orientation,
                labelpad=labelpad, lines=lines)

    def plot_beam(self, header, bmin=None, bmaj=None, bpa=0., dx=1,
            dy=1, pad=2, color='k', **kwargs):
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
                zorder=4, fc=color, **kwargs)
        #ax.add_patch(rect)
        self.ax.add_patch(beam)

    def set_xlim(self, xmin=None, xmax=None):
        self.ax.set_xlim(xmin, xmax)

    def set_ylim(self, ymin=None, ymax=None):
        self.ax.set_ylim(ymin, ymax)

    def set_title(self, title):
        self.ax.set_title(title)

    def scatter(self, x, y, **kwargs):
        #return self.ax.scatter(x, y, transform=self.ax.get_transform('world'), **kwargs)
        return super(MapPlotter, self).scatter(x, y,
                transform=self.ax.get_transform('world'), **kwargs)

    def circle(self, x, y, r, color='g', facecolor='none', zorder=0):
        cir = SphericalCircle((x, y), r, edgecolor=color, facecolor=facecolor,
                transform=self.ax.get_transform('fk5'), zorder=zorder)
        self.ax.add_patch(cir)

    def rectangle(self, blc, width, height, edgecolor='green',
            facecolor='none', **kwargs):
        r = Rectangle(blc, width, height, edgecolor=edgecolor,
                facecolor=facecolor, **kwargs)
        self.ax.add_patch(r)

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

    def plot_markers(self, markers, skip_label=False, zorder=3, **kwargs):
        for m in markers:
            # Plot marker
            mp = self.scatter(m['loc'].ra.degree, m['loc'].dec.degree, c=m['color'],
                    marker=m['style'], label=m.get('legend'), zorder=zorder,
                    **kwargs)

            # Marker label
            if m.get('label') and not skip_label:
                labloc = m['labloc'].ra.degree, m['labloc'].dec.degree
                self.annotate(m['label'].strip(), xy=labloc, xytext=labloc,
                        xycoords=self.ax.get_transform('world'),
                        color=m['color'], zorder=zorder)

class MapsPlotter(BasePlotter):

    def __init__(self, config=None, section='map_plot', projection=None, 
            **kwargs):
        super(MapsPlotter, self).__init__(config=config, section=section, 
                **kwargs)
        self._projection = projection or self.config.get('projection')

    @property
    def projection(self):
        return self._projection

    def __iter__(self):
        for ax in self.axes:
            axis, cbax = self.get_axis(ax)
            yield MapPlotter(axis, cbax, config=self.config)

    def get_mapper(self, loc, vmin=None, vmax=None, a=None, stretch=None,
            projection=None, include_cbar=None):
        axis, cbax = self.get_axis(loc, projection=projection,
                include_cbar=include_cbar)
        
        # Get color stretch values
        stretch = stretch or self.get_value('stretch', 'linear', loc)
        vmin = vmin or float(self.get_value('vmin', vmin, loc))
        vmax = vmax or float(self.get_value('vmax', vmax, loc))
        a = a or float(self.get_value('a', 1000., loc))

        # Get the axis
        self.axes[loc] = MapPlotter(axis, cbax, vmin=vmin, vmax=vmax, a=a,
                stretch=stretch)

        return self.axes[loc]

    def auto_config(self, config=None, section='map_plot', legend=False, **kwargs):
        # Read new config if requested
        if config is not None:
            cfg = ConfigParser()
            cfg.read(os.path.expanduser(config))
            cfg = cfg['map_plot']
        else:
            cfg = self.config

        # Config map options
        xformat = kwargs.get('xformat', 
                cfg.get('xformat', fallback="hh:mm:ss.s"))
        yformat = kwargs.get('yformat', 
                cfg.get('yformat', fallback="dd:mm:ss"))
        tickscolors = kwargs.get('tickscolor', 
                cfg.get('tickscolor', fallback="k"))

        # Config
        for i,(loc,ax) in enumerate(self.axes.items()):
            # Labels and ticks
            xlabel = not self.sharex or \
                    (self.sharex and loc[1]==0 and loc[0]==self.shape[0]-1)
            xticks = not self.sharex or \
                    (self.sharex and loc[0]==self.shape[0]-1)
            ylabel = not self.sharey or \
                    (self.sharey and loc[1]==0 and loc[0]==self.shape[0]-1)
            yticks = not self.sharey or (self.sharey and loc[1]==0)
            
            # Ticks color
            if len(tickscolors) == 1:
                tickscolor = tickscolors
            else:
                try:
                    tickscolor = tickscolors.replace(',',' ').split()[i]
                except IndexError:
                    tickscolor = tickscolors[0]

            ax.config_map(xformat=xformat, yformat=yformat, xlabel=xlabel,
                    ylabel=ylabel, xticks=xticks, yticks=yticks, 
                    xpad=1., ypad=-0.7, tickscolor=tickscolor, xcoord='ra', ycoord='dec')

            if (legend or self.config.getboolean('legend', fallback=False)) and loc==(0,0):
                ax.legend(auto=True, loc=4, match_colors=True,
                        fancybox=self.config.getboolean('fancybox', fallback=False),
                        framealpha=self.config.getfloat('framealpha', fallback=None),
                        facecolor=self.config.get('facecolor', fallback=None))


    def init_axis(self, loc, projection='rectilinear', include_cbar=None):
        super(MapsPlotter, self).init_axis(loc, projection, include_cbar)

