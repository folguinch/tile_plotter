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

import normalizations as mynorms
from .base_plotter import BasePlotter, SinglePlotter
from .functions import get_ticks
from .utils import auto_vmin, auto_vmax, auto_levels
from .maths import quick_rms

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
        elif self.stretch=='midnorm':
            #return ImageNormalize(vmin=self.vmin, vmax=self.vmax,
            #        stretch=mynorms.MidpointStretch())
            return mynorms.MidpointNormalize(vmin=self.vmin, vmax=self.vmax)
        else:
            return ImageNormalize(vmin=self.vmin, vmax=self.vmax, 
                    stretch=LinearStretch())

    def plot_map(self, data, wcs=None, label=None, r=None, position=None, 
            extent=None, self_contours=False, levels=None, colors='w', 
            linewidths=None, mask=False, rms=None, nsigma=5., nsigmalevel=None, 
            **kwargs):

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
            self.plot_contours(data, levels=levels, rms=rms, nsigma=nsigma,
                    wcs=wcs, extent=extent, colors=colors, zorder=2, 
                    linewidths=linewidths, nsigmalevel=nsigmalevel)

    def plot_contours(self, data, levels=None, rms=None, nsigma=5., wcs=None, 
            extent=None, colors='g', zorder=0, nsigmalevel=None, **kwargs):
        if levels is None:
            try:
                levels = auto_levels(data, rms=rms, nsigma=nsigma,
                        nsigmalevel=nsigmalevel)
            except ValueError:
                return None
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
        elif len(position)==2:
            ra, dec = position
        else:
            print('WARN: Could not recenter plot')
            return
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
                    marker=m['style'], label=m.get('legend'),
                    s=m.get('size'), zorder=zorder,
                    **kwargs)

            # Marker label
            if m.get('label') and not skip_label:
                labloc = m['labloc'].ra.degree, m['labloc'].dec.degree
                self.annotate(m['label'].strip(), xy=labloc, xytext=labloc,
                        xycoords=self.ax.get_transform('world'),
                        color=m['color'], zorder=zorder)

    def set_aspect(self, *args):
        self.ax.set_aspect(*args)

    def auto_plot(self, data, config, fig, hasxlabel, hasylabel, hasxticks, 
            hasyticks, **kwargs):
        """This function only works if myConfigParser is used
        """
        for img, wcs in data:
            # From config
            levels = config.getfloatlist('levels', fallback=None)
            r = config.getquantity('radius', fallback=None)
            position = config.getskycoord('center', fallback=None)
            self_contours = config.getboolean('contours', fallback=False)
            colors = config.get('contours_color', fallback='w')
            rms = config.getquantity('rms', fallback=None)
            nsigma = config.getfloat('nsigma', fallback=5.)
            try:
                bunit = u.Unit(img.header['BUNIT'])
            except KeyError:
                bunit = None
            
            # Plot map
            if 'add_style' in config:
                with matplotlib.pyplot.style.context(config['add_style']):
                    self.plot_map(np.squeeze(img.data), wcs=wcs, r=r,
                            position=position, self_contours=self_contours,
                            levels=levels, colors=colors, mask=False, rms=rms,
                            nsigma=nsigma, nsigmalevel=None)
            else:
                self.plot_map(np.squeeze(img.data), wcs=wcs, r=r,
                        position=position, self_contours=self_contours,
                        levels=levels, colors=colors, mask=False, rms=rms,
                        nsigma=nsigma, nsigmalevel=None)

            # Beam
            if config.getboolean('plot_beam', fallback=True):
                try:
                    self.plot_beam(img.header,
                            color=config.get('beam_color',fallback='k'))
                except TypeError:
                    pass
                    #if 'BMIN' not in img.header:
                    #    self.log.warn('Beam information not in header')
            
            # Colorbar
            if fig is not None:
                cbarlabel = config.get('cbarlabel', fallback='Intensity')
                if bunit:
                    cbarlabel += ' (%s)' % (bunit.to_string('latex_inline'),)
                if config.getboolean('vcbar', fallback=False):
                    orientation = 'vertical'
                else:
                    orientation = 'horizontal'
                self.plot_cbar(fig, orientation=orientation, label=cbarlabel,
                        labelpad=config.getfloat('labelpad', fallback=10))

        # Artists
        self.auto_artists(config)
        
        # Config
        self.auto_config(config, hasxlabel, hasylabel, hasxticks, hasyticks, 
                **kwargs)

        # Label 
        if 'label' in config:
            self.label_axes(config['label'],
                    backgroundcolor=config.get('label_background', None))

    def auto_artists(self, cfg, artists=['markers', 'arcs', 'texts']):
        for artist in artists:
            if artist=='texts':
                iterover = cfg.getvalueiter(artist, sep=',')
            else:
                iterover = cfg.getvalueiter(artist, sep=',', dtype='skycoord')
            for i, art in enumerate(iterover):
                color = cfg.getvalue('%s_color' % artist, n=i, fallback='g')
                zorder = cfg.getvalue('%s_zorder' % artist, n=i, fallback=2,
                        dtype=int)
                if artist=='markers':
                    fmt = cfg.getvalue('%s_fmt' % artist, n=i, fallback='+')
                    size = cfg.getvalue('%s_size' % artist, n=i, fallback=100,
                            dtype=float)
                    mk = self.scatter(art.ra.degree, art.dec.degree, c=color,
                            marker=fmt, s=size, zorder=zorder)
                elif artist=='arcs':
                    width = cfg.getvalue('%s_width' % artist, n=i, dtype=float)
                    height = cfg.getvalue('%s_height' % artist, n=i,
                            dtype=float)
                    if width is None or height is None:
                        print('Arc artist requires width and height')
                        continue
                    kwargs = {'angle':0.0, 'theta1':0.0, 'theta2':360.0,
                            'linewidth':1, 'linestyle':'-'}
                    textopts = ['linestyle']
                    for opt in kwargs:
                        if opt in textopts:
                            kwargs[opt] = cfg.getvalue('%s_%s' % (artist, opt),
                                    n=i, fallback=kwargs[opt])
                            continue
                        kwargs[opt] = cfg.getvalue('%s_%s' % (artist, opt),
                                n=i, fallback=kwargs[opt], dtype=float)
                    mk = self.arc((art.ra.degree, art.dec.degree), width,
                            height, color=color, zorder=zorder,
                            transform=self.ax.get_transform('world'), **kwargs)
                elif artist=='texts':
                    loc = cfg.getvalue('%s_loc' % artist, n=i, sep=',')
                    if r'\n' in art:
                        art = art.replace(r'\n', '\n')
                    loc = map(float, loc.split())
                    self.label_axes(str(art), loc=loc, color=color, zorder=zorder)


    def auto_config(self, cfg, xlabel, ylabel, xticks, yticks, **kwargs):
        # Config map options
        xformat = kwargs.get('xformat', 
                cfg.get('xformat', fallback="hh:mm:ss.s"))
        yformat = kwargs.get('yformat', 
                cfg.get('yformat', fallback="dd:mm:ss"))

        # Ticks color
        tickscolor = kwargs.get('tickscolor', 
                cfg.get('tickscolor', fallback="k"))

        # Apect ratio
        self.set_aspect(1./self.ax.get_data_ratio())

        # Config
        if cfg['type']=='pvmap':
            xlim = cfg.getfloatlist('xlim', fallback=(None,None))
            ylim = cfg.getfloatlist('ylim', fallback=(None,None))
            xlabel = 'Offset (arcsec)' if xlabel else ''
            ylabel = 'Velocity (km s$^{-1}$)' if ylabel else ''
            self.config_plot(xlim=tuple(xlim), xlabel=xlabel, 
                    unset_xticks=not xticks,
                    ylim=tuple(ylim), ylabel=ylabel, 
                    unset_yticks=not yticks, 
                    tickscolor=tickscolor)
        else:
            self.config_map(xformat=xformat, yformat=yformat, xlabel=xlabel,
                    ylabel=ylabel, xticks=xticks, yticks=yticks, 
                    xpad=1., ypad=-0.7, tickscolor=tickscolor, xcoord='ra', ycoord='dec')

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

    def auto_config(self, config=None, section='map_plot', legend=False,
            dtype='intensity', **kwargs):
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
            if not self.is_init(loc):
                break

            # Labels and ticks
            xlabel, ylabel = self.has_axlabels(loc)
            xticks, yticks = self.has_ticks(loc)
            
            # Ticks color
            if len(tickscolors) == 1:
                tickscolor = tickscolors
            else:
                try:
                    tickscolor = tickscolors.replace(',',' ').split()[i]
                except IndexError:
                    tickscolor = tickscolors[0]

            if self.config.getfloat('xsize')==self.config.getfloat('ysize'):
                self.log.info('Setting equal axis aspect ratio')
                ax.set_aspect(1./ax.ax.get_data_ratio())

            if dtype=='pvmap':
                try:
                    xlim = map(float, self.get_value('xlim', (None,None), loc,
                        sep=',').split())
                except (TypeError, AttributeError):
                    xlim = (None, None)
                try:
                    ylim = map(float, self.get_value('ylim', (None,None), loc,
                        sep=',').split())
                except (TypeError, AttributeError):
                    ylim = (None, None)
                xlabel = 'Offset (arcsec)' if xlabel else ''
                ylabel = 'Velocity (km s$^{-1}$)' if ylabel else ''
                ax.config_plot(xlim=tuple(xlim), xlabel=xlabel, 
                        unset_xticks=not xticks,
                        ylim=tuple(ylim), ylabel=ylabel, 
                        unset_yticks=not yticks, 
                        tickscolor=tickscolor)

            else:
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

